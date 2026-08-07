/// <reference lib="webworker" />
import {
  advanceLiveTraining,
  applyMlpGradSums,
  buildPayload,
  createLiveTrainingState,
  finishEpochBookkeeping,
  resetLiveTraining,
  trainOneEpochWith,
  type LiveTrainingState,
} from "../train";
import { flattenWeights, trainBatch } from "../mlp";
import type { PlaygroundConfig } from "../types";
import type { FromTrainWorker, MlpTrainSnapshot, ToTrainWorker } from "./protocol";
import { createPlayLoop } from "./runPlayLoop";
import { ShardPool } from "./ShardPool";

declare const self: DedicatedWorkerGlobalScope;

let state: LiveTrainingState | null = null;
let play: ReturnType<typeof createPlayLoop> | null = null;
let pool: ShardPool | null = null;

function post(msg: FromTrainWorker): void {
  self.postMessage(msg);
}

function createGradShardWorker(): Worker {
  return new Worker(new URL("./grad-shard.js", import.meta.url), { type: "module" });
}

function buildSnapshot(s: LiveTrainingState): MlpTrainSnapshot {
  return {
    kind: "mlp",
    config: structuredClone(s.config),
    epoch: s.epoch,
    playing: s.playing,
    payload: buildPayload(s),
  };
}

async function syncPool(): Promise<void> {
  if (!state) return;
  if (!pool) {
    pool = new ShardPool({ createShardWorker: createGradShardWorker, kind: "mlp" });
  }
  await pool.init(structuredClone(state.config));
  pool.setTrainData({
    features: state.split.trainFeatures,
    labels: state.split.trainLabels,
  });
}

async function trainEpochDp(): Promise<boolean> {
  if (!state) return false;
  if (state.epoch >= state.config.epochs) {
    state.playing = false;
    return false;
  }
  if (!pool || pool.shardCount <= 1) {
    state = advanceLiveTraining({ ...state, playing: true }, 1);
    return state.epoch < state.config.epochs;
  }
  const mlp = state.mlp;
  await trainOneEpochWith(state, async (indices) => {
    const weights = flattenWeights(mlp);
    const result = await pool!.computeGrads(weights, indices);
    if (!result || result.count === 0) {
      const batch = indices.map((i) => state!.split.trainFeatures[i]!);
      const labels = indices.map((i) => state!.split.trainLabels[i]!);
      trainBatch(mlp, batch, labels);
    } else {
      applyMlpGradSums(mlp, result.grads, result.count);
    }
  });
  state = finishEpochBookkeeping(state);
  return state.epoch < state.config.epochs;
}

function ensurePlay(): ReturnType<typeof createPlayLoop> {
  if (!play) {
    play = createPlayLoop({
      trainOneEpoch: () => trainEpochDp(),
      onPaint: () => {
        if (!state) return;
        post({ type: "tick", snapshot: buildSnapshot(state) });
      },
      maxEpochsPerFrame: 1,
      paintHz: 16,
    });
  }
  return play;
}

async function handleMessage(msg: ToTrainWorker): Promise<void> {
  switch (msg.type) {
    case "init": {
      play?.stop();
      pool?.dispose();
      pool = null;
      state = createLiveTrainingState(structuredClone(msg.config as PlaygroundConfig));
      await syncPool();
      post({ type: "ready", snapshot: buildSnapshot(state) });
      break;
    }
    case "setConfig": {
      if (!state) return;
      state = {
        ...state,
        config: { ...state.config, ...(msg.patch as Partial<PlaygroundConfig>) },
      };
      post({ type: "tick", snapshot: buildSnapshot(state) });
      break;
    }
    case "rebuild": {
      play?.pause();
      const cfg = (msg.payload as PlaygroundConfig | undefined) ?? state?.config;
      if (!cfg) return;
      state = resetLiveTraining(structuredClone(cfg));
      await syncPool();
      post({ type: "tick", snapshot: buildSnapshot(state) });
      break;
    }
    case "play": {
      if (!state) return;
      if (state.epoch >= state.config.epochs) {
        state = resetLiveTraining(structuredClone(state.config));
        await syncPool();
      }
      state.playing = true;
      ensurePlay().play(msg.epochsPerSec);
      break;
    }
    case "pause": {
      play?.pause();
      if (state) {
        state.playing = false;
        post({ type: "tick", snapshot: buildSnapshot(state) });
      }
      break;
    }
    case "step": {
      if (!state) return;
      play?.pause();
      if (state.epoch >= state.config.epochs) {
        state = resetLiveTraining(structuredClone(state.config));
        await syncPool();
      }
      await trainEpochDp();
      post({ type: "tick", snapshot: buildSnapshot(state!) });
      break;
    }
    case "dispose": {
      play?.stop();
      play = null;
      pool?.dispose();
      pool = null;
      state = null;
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
