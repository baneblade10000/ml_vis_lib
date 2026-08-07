/// <reference lib="webworker" />
/**
 * Grad-shard worker: holds a replica engine, computes gradient sums only.
 * Optimizer / Adam state stays on the coordinator.
 */
import { CnnEngine, type CnnConfig } from "../cnn/engine";
import { PlaygroundEngine, type NetworkPlaygroundConfig } from "../network/engine";
import { ComputationalGraph } from "../network/graph/computational-graph";
import type { GraphSnapshot } from "../network/graph/types";
import {
  accumulateBatchGradSums,
  createMLP,
  loadWeights,
  type MLP,
} from "../mlp";
import type { PlaygroundConfig } from "../types";
import { prepareTrainingSplit } from "../train";
import type { FromGradShard, GradShardKind, ToGradShard } from "./shardProtocol";

declare const self: DedicatedWorkerGlobalScope;

let kind: GradShardKind | null = null;
let cnn: CnnEngine | null = null;
let network: PlaygroundEngine | null = null;
let mlp: MLP | null = null;
let mlpFeatures: number[][] = [];
let mlpLabels: number[] = [];

function post(msg: FromGradShard, transfer?: Transferable[]): void {
  if (transfer?.length) self.postMessage(msg, transfer);
  else self.postMessage(msg);
}

self.onmessage = (ev: MessageEvent<ToGradShard>) => {
  const msg = ev.data;
  try {
    switch (msg.type) {
      case "init": {
        kind = msg.kind;
        cnn = null;
        network = null;
        mlp = null;
        mlpFeatures = [];
        mlpLabels = [];
        if (kind === "cnn") {
          cnn = new CnnEngine(structuredClone(msg.config as CnnConfig));
        } else if (kind === "network") {
          const payload = msg.config as {
            config: NetworkPlaygroundConfig;
            graphSnapshot?: GraphSnapshot;
          };
          network = new PlaygroundEngine(structuredClone(payload.config));
          if (payload.graphSnapshot) {
            const reg = network.graph.regularization;
            network.graph = ComputationalGraph.fromSnapshot(payload.graphSnapshot, reg);
          }
        } else {
          const cfg = structuredClone(msg.config as PlaygroundConfig);
          const split = prepareTrainingSplit(cfg);
          mlpFeatures = split.trainFeatures;
          mlpLabels = split.trainLabels;
          const inputSize = mlpFeatures[0]?.length ?? cfg.featureNames.length;
          mlp = createMLP(
            inputSize,
            cfg.hiddenLayers,
            cfg.activation,
            cfg.optimizer,
            cfg.learningRate,
            cfg.seed,
          );
        }
        post({ type: "ready" });
        break;
      }
      case "setTrainData": {
        if (kind === "cnn" && cnn) {
          cnn.trainData = msg.data as CnnEngine["trainData"];
        } else if (kind === "network" && network) {
          network.trainData = msg.data as PlaygroundEngine["trainData"];
        } else if (kind === "mlp") {
          const d = msg.data as { features: number[][]; labels: number[] };
          mlpFeatures = d.features;
          mlpLabels = d.labels;
        }
        break;
      }
      case "compute": {
        if (kind === "cnn" && cnn) {
          cnn.loadParams(msg.weights);
          cnn.zeroAllGrads();
          const count = cnn.accumulateGradIndices(msg.indices);
          const grads = cnn.exportGradSums();
          post({ type: "grads", grads, count }, [grads.buffer]);
        } else if (kind === "network" && network) {
          network.loadParams(msg.weights);
          network.zeroGradAccumulators();
          const count = network.accumulateGradIndices(msg.indices);
          const grads = network.exportGradSums();
          post({ type: "grads", grads, count }, [grads.buffer]);
        } else if (kind === "mlp" && mlp) {
          loadWeights(mlp, msg.weights);
          const batch: number[][] = [];
          const labels: number[] = [];
          for (const idx of msg.indices) {
            const f = mlpFeatures[idx];
            const y = mlpLabels[idx];
            if (f && y !== undefined) {
              batch.push(f);
              labels.push(y);
            }
          }
          const { grads, count } = accumulateBatchGradSums(mlp, batch, labels);
          post({ type: "grads", grads, count }, [grads.buffer]);
        } else {
          post({ type: "error", message: "grad shard not initialized" });
        }
        break;
      }
      case "dispose": {
        cnn = null;
        network = null;
        mlp = null;
        break;
      }
      default:
        break;
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
