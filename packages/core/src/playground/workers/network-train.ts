/// <reference lib="webworker" />
import { PlaygroundEngine, type NetworkPlaygroundConfig } from "../network/engine";
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

function workerPayload(): {
  config: NetworkPlaygroundConfig;
  graphSnapshot: GraphSnapshot;
  trainData: Array<{ x: number; y: number; label: number }>;
  testData: Array<{ x: number; y: number; label: number }>;
} {
  return {
    config: structuredClone(engine!.config),
    graphSnapshot: engine!.graph.toSnapshot(),
    trainData: clonePoints(engine!.trainData),
    testData: clonePoints(engine!.testData),
  };
}

function clonePoints(points: Array<{ x: number; y: number; label: number }>) {
  return points.map((p) => ({ x: p.x, y: p.y, label: p.label }));
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
    trainData: clonePoints(e.trainData),
    testData: clonePoints(e.testData),
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
    let result: { grads: Float64Array; count: number } | null = null;
    try {
      result = await pool.computeGrads(weights, indices);
    } catch {
      // Pool was rebuilt/terminated mid-batch (add neuron while Play).
      // Finish this epoch on the coordinator so training keeps moving.
      e.zeroGradAccumulators();
      e.accumulateGradIndices(indices);
      e.applyGradSums(e.exportGradSums(), indices.length);
      continue;
    }
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

function initFromPayload(
  payload: {
    config: NetworkPlaygroundConfig;
    graphSnapshot?: GraphSnapshot;
    trainData?: Array<{ x: number; y: number; label: number }>;
    testData?: Array<{ x: number; y: number; label: number }>;
  },
  options?: { resetTraining?: boolean },
): void {
  play?.stop();
  const config = structuredClone(payload.config);
  if (!engine) {
    engine = new PlaygroundEngine(config);
  } else {
    engine.config = {
      ...config,
      networkShape: [...config.networkShape],
      enabledFeatures: { ...config.enabledFeatures },
    };
  }
  if (payload.graphSnapshot) {
    // Rebuild boundary/curve stores for the snapshot node ids — swapping the
    // graph without this leaves heatmaps keyed to the bootstrap topology and
    // neurons paint as blank white tiles.
    engine.applyTopologySnapshot(payload.graphSnapshot, {
      trainData: payload.trainData,
      testData: payload.testData,
      resetTraining: options?.resetTraining,
    });
  } else if (payload.trainData?.length) {
    engine.trainData = clonePoints(payload.trainData);
    if (payload.testData?.length) engine.testData = clonePoints(payload.testData);
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
      // Topology / dataset edits reset training; weight-only sync must not.
      const resetTraining =
        msg.reason === "topology" ||
        msg.reason === "dataset" ||
        msg.reason === "reset" ||
        msg.reason === "resetWeights" ||
        msg.reason === "mode";
      if (msg.reason === "reset" || msg.reason === "resetWeights") {
        const payload = msg.payload as
          | { config: NetworkPlaygroundConfig; graphSnapshot?: GraphSnapshot }
          | undefined;
        if (payload) initFromPayload(payload, { resetTraining: true });
        else if (engine) {
          if (msg.reason === "reset") engine.resetToInitial();
          else engine.resetWeights();
        }
      } else if (msg.payload) {
        initFromPayload(
          msg.payload as { config: NetworkPlaygroundConfig; graphSnapshot?: GraphSnapshot },
          { resetTraining },
        );
      } else if (!engine) {
        return;
      }
      await syncPool();
      if (engine) {
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
            trainData: Array.isArray(a.trainData)
              ? (a.trainData as Array<{ x: number; y: number; label: number }>)
              : undefined,
            testData: Array.isArray(a.testData)
              ? (a.testData as Array<{ x: number; y: number; label: number }>)
              : undefined,
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

/** Serialize async handlers — play/step must not race an in-flight rebuild/pool sync. */
let messageChain: Promise<void> = Promise.resolve();

self.onmessage = (ev: MessageEvent<ToTrainWorker>) => {
  messageChain = messageChain
    .then(() => handleMessage(ev.data))
    .catch((err) => {
      post({
        type: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    });
};
