/// <reference lib="webworker" />
/**
 * CNN train worker — Rust WASM owns forward/backward/grads.
 * Single in-process engine: sharding tiny 16×16 nets is pure IPC tax.
 */
import { DEFAULT_CNN_CONFIG, SIGNAL_LENGTH, imageSizeForDataset, makeImageDataset, makeSignalDataset, type CnnConfig, type CnnDatasetId2D, type CnnTrainSnapshot, type ImageExample, type SignalExample } from "@ml-vis/core/cnn";
import { type FromTrainWorker, type ToTrainWorker } from "@ml-vis/core/workers";
import init, { WasmCnnEngine } from "./pkg/cnn.js";
import wasmUrl from "./pkg/cnn_bg.wasm?url";

declare const self: DedicatedWorkerGlobalScope;

let engine: WasmCnnEngine | null = null;
let config: CnnConfig | null = null;
let galleryExamples: Array<ImageExample | SignalExample> = [];
let cachedGalleryPredictions: number[] = [];
let cachedLayers: CnnTrainSnapshot["layers"] | null = null;
let cachedLossHistory: CnnTrainSnapshot["lossHistory"] = [];
let cachedKernels: CnnTrainSnapshot["kernels"] = {};
let cachedKernels2dIn: Record<string, number[][][][]> = {};
let cachedKernels1dIn: Record<string, number[][][]> = {};
let cachedBiases: Record<string, number[]> = {};
let cachedProbability = 0.5;
let play: ReturnType<typeof createPlayLoop> | null = null;
let wasmReady: Promise<void> | null = null;

function post(msg: FromTrainWorker): void {
  self.postMessage(msg);
}

async function ensureWasm(): Promise<void> {
  if (!wasmReady) {
    wasmReady = init({ module_or_path: wasmUrl }).then(() => undefined);
  }
  await wasmReady;
}

/** Play hot path: train only (no loss-history push / no JSValue return). */
function trainEpochFast(): void {
  engine!.trainEpoch();
}

function pack2d(examples: ImageExample[]): { images: Float32Array; labels: Int32Array } {
  const n = examples.length;
  const size = examples[0]?.pixels.length ?? 16;
  const images = new Float32Array(n * size * size);
  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const ex = examples[i]!;
    labels[i] = ex.label;
    const base = i * size * size;
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        images[base + r * size + c] = ex.pixels[r]![c]!;
      }
    }
  }
  return { images, labels };
}

function syncImageSize(cfg: CnnConfig): void {
  if (cfg.mode === "2d") {
    cfg.imageSize = imageSizeForDataset(cfg.dataset as CnnDatasetId2D);
  }
}

function pack1d(examples: SignalExample[]): { signals: Float32Array; labels: Int32Array } {
  const n = examples.length;
  const signals = new Float32Array(n * SIGNAL_LENGTH);
  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const ex = examples[i]!;
    labels[i] = ex.label;
    const base = i * SIGNAL_LENGTH;
    for (let t = 0; t < SIGNAL_LENGTH; t++) {
      signals[base + t] = ex.values[t] ?? 0;
    }
  }
  return { signals, labels };
}

function pushDataToWasm(cfg: CnnConfig): void {
  if (!engine) return;
  syncImageSize(cfg);
  // Rebuild pipeline so Input H×W matches the dataset resolution.
  engine.applyConfigJson(JSON.stringify(cfg));
  const trainRatio = (cfg.percTrainData ?? 50) / 100;
  if (cfg.mode === "2d") {
    const data = makeImageDataset(cfg.dataset as never, undefined, cfg.noise);
    const split = Math.floor(data.length * trainRatio);
    const train_n = Math.max(1, Math.min(split, data.length - 1));
    const ordered = [...data.slice(0, train_n), ...data.slice(train_n)];
    galleryExamples = ordered.slice(train_n);
    const { images, labels } = pack2d(ordered);
    engine.setData2d(images, labels, ordered.length, train_n);
  } else {
    const data = makeSignalDataset(cfg.dataset as never, undefined, cfg.noise);
    const split = Math.floor(data.length * trainRatio);
    const train_n = Math.max(1, Math.min(split, data.length - 1));
    const ordered = [...data.slice(0, train_n), ...data.slice(train_n)];
    galleryExamples = ordered.slice(train_n);
    const { signals, labels } = pack1d(ordered);
    engine.setData1d(signals, labels, ordered.length, train_n);
  }
}

function wasmPlain<T>(value: unknown): T {
  // Only flatten ES6 Map roots — deep-walking feature maps was pure CPU tax.
  if (value instanceof Map) {
    const out: Record<string, unknown> = {};
    for (const [k, v] of value.entries()) out[String(k)] = wasmPlain(v);
    return out as T;
  }
  if (Array.isArray(value) && value.length > 0 && value[0] instanceof Map) {
    return value.map((v) => wasmPlain(v)) as T;
  }
  return value as T;
}

function buildSnapshot(mode: "full" | "play-maps" | "play-maps-k"): CnnTrainSnapshot {
  const e = engine!;
  const cfg = config!;
  if (mode === "full") e.refreshMetrics();
  const stats = wasmPlain<CnnTrainSnapshot["stats"]>(e.stats());
  cachedLossHistory = wasmPlain<CnnTrainSnapshot["lossHistory"]>(e.lossHistory());
  if (mode === "full" || !cachedLayers) {
    cachedLayers = wasmPlain<CnnTrainSnapshot["layers"]>(e.metas());
  }
  const layers = cachedLayers;
  const wantKernels = mode === "full" || mode === "play-maps-k";
  const featureMaps = wasmPlain<CnnTrainSnapshot["featureMaps"]>(
    e.featureMaps(wantKernels),
  );
  // Re-attach cached kernels so conv tiles don't blank between full dumps.
  if (!wantKernels) {
    for (const m of featureMaps) {
      const k = cachedKernels[m.layerId];
      if (k?.length) {
        const first = k[0] as number[] | number[][] | undefined;
        if (Array.isArray(first) && Array.isArray(first[0])) m.kernels2d = k as number[][][];
        else m.kernels1d = k as number[][];
      }
      if (cachedKernels2dIn[m.layerId]?.length) m.kernels2dIn = cachedKernels2dIn[m.layerId];
      if (cachedKernels1dIn[m.layerId]?.length) m.kernels1dIn = cachedKernels1dIn[m.layerId];
      if (!m.biases?.length && cachedBiases[m.layerId]?.length) {
        m.biases = cachedBiases[m.layerId];
      }
    }
  }
  if (mode === "full" || cachedGalleryPredictions.length === 0) {
    cachedGalleryPredictions = wasmPlain<number[]>(e.predictGallery(48)).slice(
      0,
      Math.min(48, galleryExamples.length),
    );
  }
  const gallerySlice = galleryExamples.slice(0, 48);
  if (wantKernels) {
    const kernels: CnnTrainSnapshot["kernels"] = {};
    const kin2: Record<string, number[][][][]> = {};
    const kin1: Record<string, number[][][]> = {};
    for (const m of featureMaps) {
      if (m.kernels2d && m.kernels2d.length > 0) kernels[m.layerId] = m.kernels2d;
      else if (m.kernels1d && m.kernels1d.length > 0) kernels[m.layerId] = m.kernels1d;
      if (m.kernels2dIn && m.kernels2dIn.length > 0) kin2[m.layerId] = m.kernels2dIn;
      if (m.kernels1dIn && m.kernels1dIn.length > 0) kin1[m.layerId] = m.kernels1dIn;
    }
    cachedKernels = kernels;
    cachedKernels2dIn = kin2;
    cachedKernels1dIn = kin1;
  }
  for (const m of featureMaps) {
    if (m.biases && m.biases.length > 0) cachedBiases[m.layerId] = m.biases;
    else if (cachedBiases[m.layerId]) m.biases = cachedBiases[m.layerId];
  }
  cachedProbability = e.probability();
  return {
    kind: "cnn",
    config: cfg,
    stats,
    lossHistory: cachedLossHistory,
    layers,
    featureMaps,
    kernels: cachedKernels,
    galleryExamples: gallerySlice,
    galleryPredictions: cachedGalleryPredictions,
    inspectedExampleIndex: e.inspected(),
    loss: stats.lossTest,
    probability: cachedProbability,
  };
}

function invalidateGalleryCache(): void {
  cachedGalleryPredictions = [];
  cachedLayers = null;
  cachedLossHistory = [];
  cachedKernels = {};
  cachedKernels2dIn = {};
  cachedKernels1dIn = {};
  cachedBiases = {};
}

/** Rough MAC proxy — default ~300, 12×9×9 both layers ~12k. */
function convCost(cfg: CnnConfig): number {
  let cost = 0;
  let inC = 1;
  const spatial = (cfg.imageSize ?? 16) ** 2;
  for (const spec of cfg.layers) {
    if (spec.kind === "conv2d") {
      const k = spec.kernelSize ?? 3;
      const f = spec.filters ?? 4;
      cost += f * inC * k * k * spatial;
      inC = f;
    } else if (spec.kind === "conv1d") {
      const k = spec.kernelSize ?? 3;
      const f = spec.filters ?? 4;
      cost += f * inC * k * 64;
      inC = f;
    } else if (spec.kind === "pool2d" || spec.kind === "pool1d") {
      /* spatial shrink ignored for proxy */
    } else if (spec.kind === "gap2d" || spec.kind === "gap1d" || spec.kind === "flatten") {
      inC = 1;
    }
  }
  return cost;
}

function createPlayLoop(options: {
  trainOneEpoch: () => void;
  onPaint: (withAccuracy: boolean) => void;
  maxEpochsPerFrame?: number;
  paintHz?: number;
  accuracyEvery?: number;
}) {
  const maxEpochsPerFrame = options.maxEpochsPerFrame ?? 12;
  const paintIntervalMs = 1000 / (options.paintHz ?? 15);
  const accuracyEvery = options.accuracyEvery ?? 6;
  let playing = false;
  let epochsPerSec = 96;
  let raf = 0;
  let lastTime = 0;
  let lastPaint = 0;
  let epochBank = 0;
  let paintGen = 0;
  let emaEpochMs = 2;

  const frame = (now: number) => {
    if (!playing) return;
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;
    // Leave headroom for feature-map paint every frame interval.
    const budget = Math.max(
      1,
      Math.min(maxEpochsPerFrame, Math.floor(10 / Math.max(emaEpochMs, 0.25))),
    );
    epochBank = Math.min(epochBank + dt * epochsPerSec, budget);
    const steps = Math.floor(epochBank);
    if (steps > 0) {
      epochBank -= steps;
      const t0 = performance.now();
      for (let i = 0; i < steps; i++) {
        options.trainOneEpoch();
        // One curve point per epoch so the chart doesn't freeze between paints.
        engine!.pushLossHistory();
      }
      emaEpochMs = emaEpochMs * 0.85 + ((performance.now() - t0) / steps) * 0.15;
    }
    // Paint on its own clock — never wait for a "heavy" slot.
    if (performance.now() - lastPaint >= paintIntervalMs) {
      paintGen++;
      options.onPaint(paintGen % accuracyEvery === 0);
      lastPaint = performance.now();
    }
    raf = self.requestAnimationFrame(frame);
  };

  return {
    play(rate: number) {
      epochsPerSec = rate;
      if (!playing) {
        playing = true;
        lastTime = performance.now();
        lastPaint = 0; // force immediate first paint with maps
        raf = self.requestAnimationFrame(frame);
      }
    },
    pause() {
      playing = false;
      if (raf) self.cancelAnimationFrame(raf);
      raf = 0;
    },
    stop() {
      playing = false;
      if (raf) self.cancelAnimationFrame(raf);
      raf = 0;
    },
  };
}

function ensurePlay() {
  if (!play) {
    const fat = config ? convCost(config) > 4000 : false;
    play = createPlayLoop({
      trainOneEpoch: () => trainEpochFast(),
      onPaint: (withAccuracy) => {
        if (withAccuracy) engine!.refreshAccuracy();
        post({
          type: "tick",
          // Periodically refresh kernels too; every frame still gets live maps.
          snapshot: buildSnapshot(withAccuracy ? "play-maps-k" : "play-maps"),
        });
      },
      maxEpochsPerFrame: fat ? 6 : 24,
      paintHz: fat ? 10 : 14,
      accuracyEvery: fat ? 8 : 5,
    });
  }
  return play;
}

function handleCommand(name: string, args: unknown): void {
  const a = (args ?? {}) as Record<string, unknown>;
  const cfg = config!;
  switch (name) {
    case "setLearningRate":
      cfg.learningRate = a.lr as number;
      break;
    case "setBatchSize":
      cfg.batchSize = a.bs as number;
      break;
    case "setOptimizer":
      cfg.optimizer = a.optimizer as CnnConfig["optimizer"];
      break;
    case "setRegularization":
      cfg.regularization = a.regularization as CnnConfig["regularization"];
      break;
    case "setRegularizationRate":
      cfg.regularizationRate = a.rate as number;
      break;
    case "setActivation":
      cfg.activation = a.activation as CnnConfig["activation"];
      for (const spec of cfg.layers) {
        if (spec.kind === "conv2d" || spec.kind === "conv1d") spec.activation = cfg.activation;
      }
      break;
    case "setDataset":
      cfg.dataset = a.dataset as CnnConfig["dataset"];
      break;
    case "updateDataParams":
      if (a.noise !== undefined) cfg.noise = a.noise as number;
      if (a.percTrainData !== undefined) cfg.percTrainData = a.percTrainData as number;
      break;
    case "regenerateData":
      break;
    case "removeLayer":
      cfg.layers.splice(a.index as number, 1);
      break;
    case "setLayerFilters": {
      const spec = cfg.layers[a.index as number];
      if (spec && (spec.kind === "conv2d" || spec.kind === "conv1d")) spec.filters = a.filters as number;
      break;
    }
    case "setLayerKernelSize": {
      const spec = cfg.layers[a.index as number];
      if (spec && (spec.kind === "conv2d" || spec.kind === "conv1d")) {
        spec.kernelSize = a.kernelSize as number;
      }
      break;
    }
    case "setLayerUnits": {
      const spec = cfg.layers[a.index as number];
      if (spec && spec.kind === "dense") spec.units = a.units as number;
      break;
    }
    case "setLayerPoolKind": {
      const spec = cfg.layers[a.index as number];
      if (spec && (spec.kind === "pool2d" || spec.kind === "pool1d")) {
        spec.poolKind = a.poolKind as "max" | "avg";
      }
      break;
    }
    case "setInspectedExample":
      engine!.setInspected(a.index as number);
      return;
    default:
      throw new Error(`Unknown CNN command: ${name}`);
  }
  engine!.applyConfigJson(JSON.stringify(cfg));
  if (
    name === "setDataset" ||
    name === "updateDataParams" ||
    name === "regenerateData" ||
    name === "setActivation" ||
    name.startsWith("setLayer") ||
    name === "removeLayer"
  ) {
    pushDataToWasm(cfg);
    // Rebuild play loop so fat-net paint/epoch budget picks up new topology.
    play?.stop();
    play = null;
  }
}

async function handleMessage(msg: ToTrainWorker): Promise<void> {
  await ensureWasm();
  switch (msg.type) {
    case "init": {
      play?.stop();
      invalidateGalleryCache();
      config = structuredClone(msg.config as CnnConfig);
      syncImageSize(config);
      engine?.free();
      engine = await WasmCnnEngine.create(JSON.stringify(config));
      console.info("[cnn] backend:", engine.backend());
      pushDataToWasm(config);
      post({ type: "ready", snapshot: buildSnapshot("full") });
      break;
    }
    case "setConfig": {
      if (!engine || !config) return;
      Object.assign(config, msg.patch);
      engine.applyConfigJson(JSON.stringify(config));
      invalidateGalleryCache();
      post({ type: "tick", snapshot: buildSnapshot("full") });
      break;
    }
    case "rebuild": {
      if (!engine || !config) return;
      play?.pause();
      invalidateGalleryCache();
      if (msg.reason === "reset") {
        config = structuredClone(
          config.mode === "1d" ? DEFAULT_CNN_CONFIG["1d"] : DEFAULT_CNN_CONFIG["2d"],
        );
        engine.free();
        engine = await WasmCnnEngine.create(JSON.stringify(config));
        pushDataToWasm(config);
      } else if (msg.reason === "resetWeights") {
        engine.rebuildModel();
        engine.refreshMetrics();
        engine.pushLossHistory();
      } else if (msg.reason === "mode" && msg.payload) {
        const mode = msg.payload as "2d" | "1d";
        config = structuredClone(DEFAULT_CNN_CONFIG[mode]);
        engine.free();
        engine = await WasmCnnEngine.create(JSON.stringify(config));
        pushDataToWasm(config);
      } else if (msg.reason === "dataset" && msg.payload) {
        config.dataset = msg.payload as CnnConfig["dataset"];
        pushDataToWasm(config);
      }
      {
        const snapshot = buildSnapshot("full");
        post({ type: "tick", snapshot });
        post({ type: "rebuilt", reason: msg.reason, snapshot });
      }
      break;
    }
    case "play": {
      if (!engine) return;
      ensurePlay().play(msg.epochsPerSec);
      break;
    }
    case "pause": {
      play?.pause();
      if (engine) {
        engine.pushLossHistory();
        invalidateGalleryCache();
        post({ type: "tick", snapshot: buildSnapshot("full") });
      }
      break;
    }
    case "step": {
      if (!engine) return;
      play?.pause();
      await trainEpochFast();
      engine.pushLossHistory();
      invalidateGalleryCache();
      post({ type: "tick", snapshot: buildSnapshot("full") });
      break;
    }
    case "inspect": {
      if (!engine) return;
      if (msg.exampleIndex !== undefined) engine.setInspected(msg.exampleIndex);
      post({ type: "tick", snapshot: buildSnapshot("play-maps") });
      break;
    }
    case "command": {
      if (!engine || !config) return;
      handleCommand(msg.name, msg.args);
      invalidateGalleryCache();
      post({ type: "tick", snapshot: buildSnapshot("full") });
      break;
    }
    case "dispose": {
      play?.stop();
      play = null;
      engine?.free();
      engine = null;
      config = null;
      invalidateGalleryCache();
      break;
    }
  }
}

self.onmessage = (ev: MessageEvent<ToTrainWorker>) => {
  void handleMessage(ev.data).catch((err) => {
    post({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  });
};
