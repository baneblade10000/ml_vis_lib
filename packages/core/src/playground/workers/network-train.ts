/// <reference lib="webworker" />
import { PlaygroundEngine, type NetworkPlaygroundConfig } from "../network/engine";
import { ComputationalGraph } from "../network/graph/computational-graph";
import type { GraphSnapshot } from "../network/graph/types";
import type { FromTrainWorker, NetworkTrainSnapshot, ToTrainWorker } from "./protocol";
import { createPlayLoop } from "./runPlayLoop";
import { ShardPool } from "./ShardPool";

declare const self: DedicatedWorkerGlobalScope;

let engine: PlaygroundEngine | null = null;
let play: ReturnType<typeof createPlayLoop> | null = null;
let pool: ShardPool | null = null;

function post(msg: FromTrainWorker): void {
  self.postMessage(msg);
}

function createGradShardWorker(): Worker {
  return new Worker(new URL("./grad-shard.js", import.meta.url), { type: "module" });
}

function workerPayload(): { config: NetworkPlaygroundConfig; graphSnapshot: GraphSnapshot } {
  return {
    config: structuredClone(engine!.config),
    graphSnapshot: engine!.graph.toSnapshot(),
  };
}

function buildSnapshot(e: PlaygroundEngine, fullBoundary: boolean): NetworkTrainSnapshot {
  if (fullBoundary) {
    e.refreshMetrics();
    e.refreshBoundary();
  } else {
    e.lossTrain = e.getLoss(e.trainData);
    e.lossTest = e.getLoss(e.testData);
    e.refreshOutputBoundaryFast();
    e.refreshHiddenBoundariesFast();
  }
  e.pushLossHistory();
  return {
    kind: "network",
    config: structuredClone(e.config),
    epoch: e.epoch,
    lossTrain: e.lossTrain,
    lossTest: e.lossTest,
    lossHistory: e.lossHistory.map((p) => ({ ...p })),
    boundary: e.boundary,
    curves: e.curves,
    targetCurve: e.targetCurve,
    graphSnapshot: e.graph.toSnapshot(),
    trainData: e.trainData.map((p) => ({ x: p.x, y: p.y, label: p.label })),
  };
}

async function syncPool(): Promise<void> {
  if (!engine) return;
  if (!pool) {
    pool = new ShardPool({ createShardWorker: createGradShardWorker, kind: "network" });
  }
  await pool.init(workerPayload());
  pool.setTrainData(engine.trainData.map((p) => ({ x: p.x, y: p.y, label: p.label })));
}

async function trainEpochDp(): Promise<void> {
  const e = engine!;
  const { batchSize } = e.config;
  const n = e.trainData.length;
  if (!pool || pool.shardCount <= 1 || n === 0) {
    e.trainEpoch(false);
    return;
  }
  e.epoch++;
  for (let start = 0; start < n; start += batchSize) {
    const end = Math.min(start + batchSize, n);
    // Match classic playground: skip trailing partial batch.
    if (end - start < batchSize && start > 0) break;
    if (end - start === 0) break;
    const indices: number[] = [];
    for (let i = start; i < end; i++) indices.push(i);
    const weights = e.flattenParams();
    const result = await pool.computeGrads(weights, indices);
    if (!result || result.count === 0) {
      e.zeroGradAccumulators();
      e.accumulateGradIndices(indices);
      e.applyGradSums(e.exportGradSums(), indices.length);
    } else {
      e.applyGradSums(result.grads, result.count);
    }
  }
}

function ensurePlay(): ReturnType<typeof createPlayLoop> {
  if (!play) {
    play = createPlayLoop({
      trainOneEpoch: () => trainEpochDp(),
      onPaint: () => {
        if (!engine) return;
        post({ type: "tick", snapshot: buildSnapshot(engine, false) });
      },
      maxEpochsPerFrame: 2,
      paintHz: 30,
    });
  }
  return play;
}

function initFromPayload(payload: {
  config: NetworkPlaygroundConfig;
  graphSnapshot?: GraphSnapshot;
}): void {
  play?.stop();
  engine = new PlaygroundEngine(structuredClone(payload.config));
  if (payload.graphSnapshot) {
    const reg = engine.graph.regularization;
    engine.graph = ComputationalGraph.fromSnapshot(payload.graphSnapshot, reg);
    engine.refreshMetrics();
    engine.refreshBoundary();
  }
}

async function handleMessage(msg: ToTrainWorker): Promise<void> {
  switch (msg.type) {
    case "init": {
      pool?.dispose();
      pool = null;
      initFromPayload(msg.config as { config: NetworkPlaygroundConfig; graphSnapshot?: GraphSnapshot });
      await syncPool();
      post({ type: "ready", snapshot: buildSnapshot(engine!, true) });
      break;
    }
    case "setConfig": {
      if (!engine) return;
      Object.assign(engine.config, msg.patch);
      if ("optimizer" in msg.patch) engine.setOptimizer(msg.patch.optimizer as never);
      post({ type: "tick", snapshot: buildSnapshot(engine, false) });
      break;
    }
    case "rebuild": {
      play?.pause();
      if (msg.reason === "reset" || msg.reason === "resetWeights") {
        const payload = msg.payload as
          | { config: NetworkPlaygroundConfig; graphSnapshot?: GraphSnapshot }
          | undefined;
        if (payload) initFromPayload(payload);
        else if (engine) {
          if (msg.reason === "reset") engine.resetToInitial();
          else engine.resetWeights();
        }
      } else if (msg.payload) {
        initFromPayload(msg.payload as { config: NetworkPlaygroundConfig; graphSnapshot?: GraphSnapshot });
      } else if (!engine) {
        return;
      }
      await syncPool();
      if (engine) post({ type: "tick", snapshot: buildSnapshot(engine, true) });
      break;
    }
    case "play": {
      if (!engine) return;
      ensurePlay().play(msg.epochsPerSec);
      break;
    }
    case "pause": {
      play?.pause();
      if (engine) post({ type: "tick", snapshot: buildSnapshot(engine, true) });
      break;
    }
    case "step": {
      if (!engine) return;
      play?.pause();
      await trainEpochDp();
      engine.refreshMetrics();
      engine.refreshBoundary();
      engine.pushLossHistory();
      post({ type: "tick", snapshot: buildSnapshot(engine, true) });
      break;
    }
    case "command": {
      if (!engine) return;
      const a = (msg.args ?? {}) as Record<string, unknown>;
      switch (msg.name) {
        case "setLearningRate":
          engine.config.learningRate = a.lr as number;
          break;
        case "setOptimizer":
          engine.config.optimizer = a.optimizer as never;
          engine.optStep = 0;
          break;
        case "setActivation":
          engine.setActivation(a.activation as never);
          break;
        case "setRegularization":
          engine.setRegularization(a.regularization as never);
          break;
        case "setRegularizationRate":
          engine.config.regularizationRate = a.rate as number;
          break;
        case "setBatchSize":
          engine.config.batchSize = a.bs as number;
          break;
        case "syncGraph":
          initFromPayload({
            config: engine.config,
            graphSnapshot: a.graphSnapshot as GraphSnapshot,
          });
          await syncPool();
          break;
        default:
          throw new Error(`Unknown network command: ${msg.name}`);
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
      break;
    }
    default:
      break;
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
