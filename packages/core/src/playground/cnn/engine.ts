/**
 * CNN playground config + snapshot types.
 * Training / forward / backward live in Burn WASM (`packages/playground` burn worker).
 */
import type { CnnDatasetId2D, CnnDatasetId1D } from "./gallery";
import type { CnnActivationId } from "./activations";
import type { CnnRegularizationId } from "./regularization";
import type { PlaygroundOptimizerId } from "../optimizers";

export type { CnnRegularizationId } from "./regularization";
export { CNN_REGULARIZATIONS, CNN_REGULARIZATION_RATES } from "./regularization";
export type { PlaygroundOptimizerId } from "../optimizers";
export { PLAYGROUND_OPTIMIZERS } from "../optimizers";

export type CnnMode = "2d" | "1d";

export type PoolKind = "max" | "avg";

/** Declarative spec for one layer — the editable description of the network. */
export interface LayerSpec {
  kind: "conv2d" | "pool2d" | "conv1d" | "pool1d" | "gap2d" | "gap1d" | "flatten" | "dense";
  filters?: number;
  kernelSize?: number;
  poolKind?: PoolKind;
  units?: number;
  activation?: CnnActivationId;
}

export interface CnnConfig {
  mode: CnnMode;
  /** Dataset id (must match `mode`). */
  dataset: CnnDatasetId2D | CnnDatasetId1D;
  /** Ordered layer specs (input + output are implicit). */
  layers: LayerSpec[];
  learningRate: number;
  optimizer: PlaygroundOptimizerId;
  activation: CnnActivationId;
  batchSize: number;
  /** Noise level ∈ [0,1] passed to the dataset generator. */
  noise: number;
  percTrainData: number;
  regularization: CnnRegularizationId;
  regularizationRate: number;
  /**
   * Spatial size of 2-D inputs (H=W). Defaults to 16; `three-four-loops` uses 32.
   * Synced from the dataset in the train worker.
   */
  imageSize?: number;
}

export const DEFAULT_CNN_CONFIG_2D: CnnConfig = {
  mode: "2d",
  dataset: "digits",
  layers: [
    { kind: "conv2d", filters: 4, kernelSize: 3, activation: "relu" },
    { kind: "pool2d", poolKind: "max" },
    { kind: "conv2d", filters: 8, kernelSize: 3, activation: "relu" },
    { kind: "gap2d" },
    { kind: "dense", units: 1, activation: "linear" },
  ],
  learningRate: 0.1,
  optimizer: "SGD",
  activation: "relu",
  batchSize: 16,
  noise: 0.1,
  percTrainData: 50,
  regularization: "none",
  regularizationRate: 0,
};

export const DEFAULT_CNN_CONFIG_1D: CnnConfig = {
  mode: "1d",
  dataset: "heartbeat",
  layers: [
    { kind: "conv1d", filters: 4, kernelSize: 5, activation: "relu" },
    { kind: "pool1d", poolKind: "max" },
    { kind: "conv1d", filters: 8, kernelSize: 5, activation: "relu" },
    { kind: "gap1d" },
    { kind: "dense", units: 1, activation: "linear" },
  ],
  learningRate: 0.1,
  optimizer: "SGD",
  activation: "relu",
  batchSize: 16,
  noise: 0.1,
  percTrainData: 50,
  regularization: "none",
  regularizationRate: 0,
};

export const DEFAULT_CNN_CONFIG: Record<CnnMode, CnnConfig> = {
  "2d": DEFAULT_CNN_CONFIG_2D,
  "1d": DEFAULT_CNN_CONFIG_1D,
};

export interface TrainingStats {
  epoch: number;
  lossTrain: number;
  lossTest: number;
  accTrain: number;
  accTest: number;
}

export interface FeatureMapSnapshot {
  layerId: string;
  /** For 2-D layers: one map per channel. For 1-D layers: empty. */
  maps2d?: number[][][];
  /** For 1-D layers: one row per channel. */
  signals?: number[][];
  /** Dense weight matrix `W[out][in]` for matrix visualization. */
  matrix?: number[][];
  /**
   * Conv2D display kernels: one `k×k` map per out filter (sum over in channels).
   * Index-aligned with {@link maps2d}.
   */
  kernels2d?: number[][][];
  /**
   * Conv2D per-input-channel kernels: `[outFilter][inChannel][ky][kx]`.
   * Present when the layer takes more than one input channel.
   */
  kernels2dIn?: number[][][][];
  /**
   * Conv1D display kernels: one length-`k` vector per out filter (sum over in channels).
   * Index-aligned with {@link signals}.
   */
  kernels1d?: number[][];
  /** Conv1D per-input-channel kernels: `[outFilter][inChannel][k]`. */
  kernels1dIn?: number[][][];
  /**
   * Per-filter biases for conv layers (index-aligned with kernels / feature maps).
   */
  biases?: number[];
}
