/// <reference lib="webworker" />
import { CnnEngine, type CnnConfig } from "../cnn/engine";
import type { CnnLayerView, CnnTrainSnapshot, FromTrainWorker, ToTrainWorker } from "./protocol";
import { createPlayLoop } from "./runPlayLoop";
import { ShardPool } from "./ShardPool";

declare const self: DedicatedWorkerGlobalScope;

let engine: CnnEngine | null = null;
let play: ReturnType<typeof createPlayLoop> | null = null;
let pool: ShardPool | null = null;

function post(msg: FromTrainWorker): void {
  self.postMessage(msg);
}

function createGradShardWorker(): Worker {
  return new Worker(new URL("./grad-shard.js", import.meta.url), { type: "module" });
}

function buildSnapshot(e: CnnEngine): CnnTrainSnapshot {
  e.refreshMetrics();
  e.forwardInspected();
  const shapes = e.pipelineShapes();
  const layers: CnnLayerView[] = e.layers.map((layer, idx) => ({
    id: layer.id,
    kind: layer.kind,
    label: layer.label(),
    shape: shapes[idx]!,
    params: layer.paramCount(),
    weightMag: layer.weightMagnitude(),
  }));
  const featureMaps = e.featureMaps();
  const gallerySource = (e.testData.length ? e.testData : e.trainData).slice();
  const galleryExamples = gallerySource.slice(0, 48);
  const galleryPredictions = galleryExamples.map((ex) => e.predict(ex));
  const data = e.trainData;
  const idx = Math.min(e.inspectedExampleIndex, Math.max(0, data.length - 1));
  const probability = data.length ? e.predict(data[idx]!) : 0.5;
  return {
    kind: "cnn",
    config: structuredClone(e.config),
    stats: e.stats(),
    lossHistory: e.lossHistory.map((p) => ({ ...p })),
    layers,
    featureMaps,
    kernels: e.kernelSnapshots(),
    galleryExamples,
    galleryPredictions,
    inspectedExampleIndex: e.inspectedExampleIndex,
    loss: e.lossTest,
    probability,
  };
}

async function syncPool(): Promise<void> {
  if (!engine) return;
  if (!pool) {
    pool = new ShardPool({ createShardWorker: createGradShardWorker, kind: "cnn" });
  }
  await pool.init(structuredClone(engine.config));
  pool.setTrainData(engine.trainData);
}

async function trainEpochDp(): Promise<void> {
  const e = engine!;
  const { batchSize, learningRate } = e.config;
  const n = e.trainData.length;
  if (!pool || pool.shardCount <= 1 || n === 0) {
    e.trainEpoch();
    return;
  }
  for (let start = 0; start < n; start += batchSize) {
    const indices: number[] = [];
    for (let i = start; i < Math.min(start + batchSize, n); i++) indices.push(i);
    const weights = e.flattenParams();
    const result = await pool.computeGrads(weights, indices);
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
        post({ type: "tick", snapshot: buildSnapshot(engine) });
      },
      maxEpochsPerFrame: 2,
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
      engine = new CnnEngine(structuredClone(msg.config as CnnConfig));
      await syncPool();
      post({ type: "ready", snapshot: buildSnapshot(engine) });
      break;
    }
    case "setConfig": {
      if (!engine) return;
      Object.assign(engine.config, msg.patch);
      post({ type: "tick", snapshot: buildSnapshot(engine) });
      break;
    }
    case "rebuild": {
      if (!engine) return;
      play?.pause();
      if (msg.reason === "reset") engine.reset();
      else if (msg.reason === "resetWeights") engine.resetWeights();
      else if (msg.reason === "mode" && msg.payload) engine.setMode(msg.payload as never);
      else if (msg.reason === "dataset" && msg.payload) engine.setDataset(msg.payload as never);
      await syncPool();
      post({ type: "tick", snapshot: buildSnapshot(engine) });
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
        post({ type: "tick", snapshot: buildSnapshot(engine) });
      }
      break;
    }
    case "step": {
      if (!engine) return;
      play?.pause();
      await trainEpochDp();
      engine.pushLossHistory();
      post({ type: "tick", snapshot: buildSnapshot(engine) });
      break;
    }
    case "inspect": {
      if (!engine) return;
      if (msg.exampleIndex !== undefined) {
        engine.inspectedExampleIndex = msg.exampleIndex;
      }
      engine.forwardInspected();
      post({ type: "tick", snapshot: buildSnapshot(engine) });
      break;
    }
    case "command": {
      if (!engine) return;
      handleCommand(msg.name, msg.args);
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
      post({ type: "tick", snapshot: buildSnapshot(engine) });
      break;
    }
    case "dispose": {
      play?.stop();
      play = null;
      pool?.dispose();
      pool = null;
      engine = null;
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
