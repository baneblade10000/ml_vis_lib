/// <reference lib="webworker" />
import { CnnEngine, type CnnConfig } from "../cnn/engine";
import type { ImageExample, SignalExample } from "../cnn/gallery";
import type { CnnLayerView, CnnTrainSnapshot, FromTrainWorker, ToTrainWorker } from "./protocol";
import { createPlayLoop } from "./runPlayLoop";
import { ShardPool } from "./ShardPool";
import { defaultShardCount } from "./shardCount";

declare const self: DedicatedWorkerGlobalScope;

let engine: CnnEngine | null = null;
let play: ReturnType<typeof createPlayLoop> | null = null;
let pool: ShardPool | null = null;

/** Cached across play ticks so we don't re-predict the gallery at paintHz. */
let cachedGalleryExamples: Array<ImageExample | SignalExample> = [];
let cachedGalleryPredictions: number[] = [];
let cachedLayers: CnnLayerView[] | null = null;
let cachedConfig: CnnConfig | null = null;

function post(msg: FromTrainWorker): void {
  self.postMessage(msg);
}

function createGradShardWorker(): Worker {
  return new Worker(new URL("./grad-shard.js", import.meta.url), { type: "module" });
}

function buildLayerViews(e: CnnEngine): CnnLayerView[] {
  const shapes = e.pipelineShapes();
  return e.layers.map((layer, idx) => ({
    id: layer.id,
    kind: layer.kind,
    label: layer.label(),
    shape: shapes[idx]!,
    params: layer.paramCount(),
    weightMag: layer.weightMagnitude(),
  }));
}

function kernelsFromFeatureMaps(
  featureMaps: ReturnType<CnnEngine["snapshotFeatureMaps"]>,
): Record<string, number[][] | number[][][]> {
  const kernels: Record<string, number[][] | number[][][]> = {};
  for (const m of featureMaps) {
    if (m.kernels2d) kernels[m.layerId] = m.kernels2d;
    else if (m.kernels1d) kernels[m.layerId] = m.kernels1d;
  }
  return kernels;
}

/**
 * @param full — pause/step/init: full metrics + gallery. Play ticks use a cheap path.
 */
function buildSnapshot(e: CnnEngine, full: boolean): CnnTrainSnapshot {
  if (full) {
    e.refreshMetrics();
  } else {
    e.refreshMetricsSampled(24);
  }
  e.forwardInspected();
  const featureMaps = e.snapshotFeatureMaps();
  const kernels = kernelsFromFeatureMaps(featureMaps);

  if (full || cachedGalleryExamples.length === 0) {
    const gallerySource = (e.testData.length ? e.testData : e.trainData).slice();
    cachedGalleryExamples = gallerySource.slice(0, 48);
    cachedGalleryPredictions = cachedGalleryExamples.map((ex) => e.predict(ex));
  }

  if (full || !cachedLayers || !cachedConfig) {
    cachedLayers = buildLayerViews(e);
    cachedConfig = structuredClone(e.config);
  } else {
    // Refresh weight magnitudes cheaply; topology is unchanged during Play.
    for (let i = 0; i < cachedLayers.length; i++) {
      const layer = e.layers[i];
      if (layer) cachedLayers[i]!.weightMag = layer.weightMagnitude();
    }
  }

  const probability = e.trainData.length || e.testData.length ? e.currentProbability() : 0.5;

  return {
    kind: "cnn",
    config: full ? structuredClone(e.config) : cachedConfig!,
    stats: e.stats(),
    lossHistory: e.lossHistory.map((p) => ({ ...p })),
    layers: cachedLayers!,
    featureMaps,
    kernels,
    galleryExamples: cachedGalleryExamples,
    galleryPredictions: cachedGalleryPredictions,
    inspectedExampleIndex: e.inspectedExampleIndex,
    loss: e.lossTest,
    probability,
  };
}

function invalidateSnapshotCache(): void {
  cachedGalleryExamples = [];
  cachedGalleryPredictions = [];
  cachedLayers = null;
  cachedConfig = null;
}

async function syncPool(): Promise<void> {
  if (!engine) return;
  if (!pool) {
    // Cap shards: tiny CNN batches make DP coordination cost more than the math.
    pool = new ShardPool({
      createShardWorker: createGradShardWorker,
      kind: "cnn",
      shardCount: Math.min(4, defaultShardCount()),
    });
  }
  await pool.init(structuredClone(engine.config));
  pool.setTrainData(engine.trainData);
}

async function trainEpochDp(): Promise<void> {
  const e = engine!;
  const { batchSize, learningRate } = e.config;
  const n = e.trainData.length;
  // Fall back to local path when shards would each get too few examples.
  const minPerShard = 4;
  if (
    !pool ||
    pool.shardCount <= 1 ||
    n === 0 ||
    batchSize < pool.shardCount * minPerShard
  ) {
    e.trainEpoch();
    return;
  }
  for (let start = 0; start < n; start += batchSize) {
    const indices: number[] = [];
    for (let i = start; i < Math.min(start + batchSize, n); i++) indices.push(i);
    const weights = e.flattenParams();
    let result: { grads: Float64Array; count: number } | null = null;
    try {
      result = await pool.computeGrads(weights, indices);
    } catch {
      e.zeroAllGrads();
      e.accumulateGradIndices(indices);
      e.applyUpdate(learningRate, indices.length);
      continue;
    }
    if (!result || result.count === 0) {
      e.zeroAllGrads();
      e.accumulateGradIndices(indices);
      e.applyUpdate(learningRate, indices.length);
    } else {
      e.loadGradSums(result.grads);
      e.applyUpdate(learningRate, result.count);
    }
  }
  e.epoch++;
}

function ensurePlay(): ReturnType<typeof createPlayLoop> {
  if (!play) {
    play = createPlayLoop({
      trainOneEpoch: () => trainEpochDp(),
      onPaint: () => {
        if (!engine) return;
        engine.pushLossHistory();
        post({ type: "tick", snapshot: buildSnapshot(engine, false) });
      },
      maxEpochsPerFrame: 2,
      // 10 Hz is plenty for feature-map viz; full metrics were the real bottleneck.
      paintHz: 10,
    });
  }
  return play;
}

function handleCommand(name: string, args: unknown): void {
  const e = engine!;
  const a = (args ?? {}) as Record<string, unknown>;
  switch (name) {
    case "setLearningRate":
      e.setLearningRate(a.lr as number);
      break;
    case "setBatchSize":
      e.setBatchSize(a.bs as number);
      break;
    case "setOptimizer":
      e.setOptimizer(a.optimizer as never);
      break;
    case "setRegularization":
      e.setRegularization(a.regularization as never);
      break;
    case "setRegularizationRate":
      e.setRegularizationRate(a.rate as number);
      break;
    case "setActivation":
      e.setActivation(a.activation as never);
      break;
    case "setDataset":
      e.setDataset(a.dataset as never);
      break;
    case "setMode":
      e.setMode(a.mode as never);
      break;
    case "updateDataParams":
      e.updateDataParams(a as { noise?: number; percTrainData?: number });
      break;
    case "regenerateData":
      e.regenerateData();
      break;
    case "removeLayer":
      e.removeLayer(a.index as number);
      break;
    case "setLayerFilters":
      e.setLayerFilters(a.index as number, a.filters as number);
      break;
    case "setLayerKernelSize": {
      const spec = e.config.layers[a.index as number];
      if (spec && (spec.kind === "conv2d" || spec.kind === "conv1d")) {
        spec.kernelSize = a.kernelSize as number;
        e.rebuildAfterSpecEdit(a.index as number);
      }
      break;
    }
    case "setLayerUnits":
      e.setLayerUnits(a.index as number, a.units as number);
      break;
    case "setLayerPoolKind": {
      const spec = e.config.layers[a.index as number];
      if (spec && (spec.kind === "pool2d" || spec.kind === "pool1d")) {
        spec.poolKind = a.poolKind as "max" | "avg";
        e.rebuildAfterSpecEdit(a.index as number);
      }
      break;
    }
    case "setInspectedExample":
      e.inspectedExampleIndex = a.index as number;
      e.forwardInspected();
      break;
    default:
      throw new Error(`Unknown CNN command: ${name}`);
  }
}

async function handleMessage(msg: ToTrainWorker): Promise<void> {
  switch (msg.type) {
    case "init": {
      play?.stop();
      pool?.dispose();
      pool = null;
      invalidateSnapshotCache();
      engine = new CnnEngine(structuredClone(msg.config as CnnConfig));
      await syncPool();
      post({ type: "ready", snapshot: buildSnapshot(engine, true) });
      break;
    }
    case "setConfig": {
      if (!engine) return;
      Object.assign(engine.config, msg.patch);
      invalidateSnapshotCache();
      post({ type: "tick", snapshot: buildSnapshot(engine, true) });
      break;
    }
    case "rebuild": {
      if (!engine) return;
      play?.pause();
      invalidateSnapshotCache();
      if (msg.reason === "reset") engine.reset();
      else if (msg.reason === "resetWeights") engine.resetWeights();
      else if (msg.reason === "mode" && msg.payload) engine.setMode(msg.payload as never);
      else if (msg.reason === "dataset" && msg.payload) engine.setDataset(msg.payload as never);
      await syncPool();
      {
        const snapshot = buildSnapshot(engine, true);
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
        invalidateSnapshotCache();
        post({ type: "tick", snapshot: buildSnapshot(engine, true) });
      }
      break;
    }
    case "step": {
      if (!engine) return;
      play?.pause();
      await trainEpochDp();
      engine.pushLossHistory();
      invalidateSnapshotCache();
      post({ type: "tick", snapshot: buildSnapshot(engine, true) });
      break;
    }
    case "inspect": {
      if (!engine) return;
      if (msg.exampleIndex !== undefined) {
        engine.inspectedExampleIndex = msg.exampleIndex;
      }
      engine.forwardInspected();
      // Keep gallery cache; only refresh maps for the new example.
      post({ type: "tick", snapshot: buildSnapshot(engine, false) });
      break;
    }
    case "command": {
      if (!engine) return;
      handleCommand(msg.name, msg.args);
      invalidateSnapshotCache();
      if (
        msg.name === "regenerateData" ||
        msg.name === "setDataset" ||
        msg.name === "setMode" ||
        msg.name === "updateDataParams" ||
        msg.name === "removeLayer" ||
        msg.name === "setLayerFilters" ||
        msg.name === "setLayerKernelSize" ||
        msg.name === "setLayerUnits" ||
        msg.name === "setLayerPoolKind"
      ) {
        await syncPool();
      } else {
        pool?.setTrainData(engine.trainData);
      }
      post({ type: "tick", snapshot: buildSnapshot(engine, true) });
      break;
    }
    case "dispose": {
      play?.stop();
      play = null;
      pool?.dispose();
      pool = null;
      engine = null;
      invalidateSnapshotCache();
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
