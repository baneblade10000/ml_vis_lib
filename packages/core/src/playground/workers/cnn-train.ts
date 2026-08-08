/// <reference lib="webworker" />
import { CnnEngine, type CnnConfig } from "../cnn/engine";
import type { ImageExample, SignalExample } from "../cnn/gallery";
import type { CnnLayerView, CnnTrainSnapshot, FromTrainWorker, ToTrainWorker } from "./protocol";
import { createPlayLoop } from "./runPlayLoop";

declare const self: DedicatedWorkerGlobalScope;

let engine: CnnEngine | null = null;
let play: ReturnType<typeof createPlayLoop> | null = null;

/** Cached across play ticks so we don't re-predict the gallery at paintHz. */
let cachedGalleryExamples: Array<ImageExample | SignalExample> = [];
let cachedGalleryPredictions: number[] = [];
let cachedLayers: CnnLayerView[] | null = null;
let cachedConfig: CnnConfig | null = null;
function post(msg: FromTrainWorker): void {
  self.postMessage(msg);
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

type SnapshotMode = "full" | "play";

/**
 * full — pause/step/init: full metrics + gallery.
 * play — feature maps; loss/acc already maintained by trainEpoch + periodic sample.
 */
function buildSnapshot(e: CnnEngine, mode: SnapshotMode): CnnTrainSnapshot {
  if (mode === "full") {
    e.refreshMetrics();
  }

  // Inspected forward first — feature maps + readout probability must match the
  // gallery thumb the user selected (not a leftover train/acc sample).
  e.forwardInspected();
  const featureMaps = e.snapshotFeatureMaps();
  const kernels = kernelsFromFeatureMaps(featureMaps);
  const probability = e.galleryData().length ? e.currentProbability() : 0.5;

  if (mode === "full" || cachedGalleryExamples.length === 0) {
    const gallerySource = e.galleryData().slice();
    cachedGalleryExamples = gallerySource.slice(0, 48);
    // predict() overwrites the live forward — restore inspected after.
    cachedGalleryPredictions = cachedGalleryExamples.map((ex) => e.predict(ex));
    e.forwardInspected();
  }

  if (mode === "full" || !cachedLayers || !cachedConfig) {
    cachedLayers = buildLayerViews(e);
    cachedConfig = structuredClone(e.config);
  }

  return {
    kind: "cnn",
    config: mode === "full" ? structuredClone(e.config) : cachedConfig!,
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

function trainOneEpoch(): void {
  engine!.trainEpoch();
}

function ensurePlay(): ReturnType<typeof createPlayLoop> {
  if (!play) {
    play = createPlayLoop({
      // Sync path — no shard pool. CNN batches are too small for DP to win.
      trainOneEpoch,
      onPaint: () => {
        if (!engine) return;
        // Toolbar accuracies — every paint (~20 Hz), small sample. Loss is per-epoch.
        engine.refreshAccuracySampled(12);
        post({ type: "tick", snapshot: buildSnapshot(engine, "play") });
      },
      // Fill the coordinator core: ~40–45 eps is the local train ceiling on this net.
      maxEpochsPerFrame: 8,
      // Higher paint rate keeps the learning curve advancing smoothly.
      paintHz: 20,
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
      e.setInspectedExample(a.index as number);
      break;
    default:
      throw new Error(`Unknown CNN command: ${name}`);
  }
}

async function handleMessage(msg: ToTrainWorker): Promise<void> {
  switch (msg.type) {
    case "init": {
      play?.stop();
      invalidateSnapshotCache();
      engine = new CnnEngine(structuredClone(msg.config as CnnConfig));
      post({ type: "ready", snapshot: buildSnapshot(engine, "full") });
      break;
    }
    case "setConfig": {
      if (!engine) return;
      Object.assign(engine.config, msg.patch);
      invalidateSnapshotCache();
      post({ type: "tick", snapshot: buildSnapshot(engine, "full") });
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
      {
        const snapshot = buildSnapshot(engine, "full");
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
        post({ type: "tick", snapshot: buildSnapshot(engine, "full") });
      }
      break;
    }
    case "step": {
      if (!engine) return;
      play?.pause();
      trainOneEpoch();
      engine.pushLossHistory();
      invalidateSnapshotCache();
      post({ type: "tick", snapshot: buildSnapshot(engine, "full") });
      break;
    }
    case "inspect": {
      if (!engine) return;
      if (msg.exampleIndex !== undefined) {
        engine.setInspectedExample(msg.exampleIndex);
      } else {
        engine.forwardInspected();
      }
      post({ type: "tick", snapshot: buildSnapshot(engine, "play") });
      break;
    }
    case "command": {
      if (!engine) return;
      handleCommand(msg.name, msg.args);
      invalidateSnapshotCache();
      post({ type: "tick", snapshot: buildSnapshot(engine, "full") });
      break;
    }
    case "dispose": {
      play?.stop();
      play = null;
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
