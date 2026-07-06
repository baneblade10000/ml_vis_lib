/** Ported from AGILAB PyTorch Playground (BSD-3-Clause, Thales). */

export type DatasetId = "circles" | "xor" | "spiral" | "gaussian";
export type ActivationId = "tanh" | "relu" | "sigmoid";
export type OptimizerId = "Adam" | "SGD";

export const DATASETS: DatasetId[] = ["circles", "xor", "spiral", "gaussian"];
export const ACTIVATIONS: ActivationId[] = ["tanh", "relu", "sigmoid"];
export const OPTIMIZERS: OptimizerId[] = ["Adam", "SGD"];

export const DEFAULT_FEATURES = ["x1", "x2"] as const;
export type FeatureName =
  | "x1"
  | "x2"
  | "x1_squared"
  | "x2_squared"
  | "x1_x2"
  | "sin_x1"
  | "sin_x2";

export interface PlaygroundConfig {
  dataset: DatasetId;
  sampleCount: number;
  noise: number;
  trainRatio: number;
  hiddenLayers: number[];
  activation: ActivationId;
  optimizer: OptimizerId;
  learningRate: number;
  epochs: number;
  batchSize: number;
  seed: number;
  featureNames: FeatureName[];
  gridSize: number;
}

export interface Sample {
  x1: number;
  x2: number;
  target: number;
}

export interface GridPoint {
  x1: number;
  x2: number;
  probability: number;
}

export interface HistoryRow {
  epoch: number;
  trainLoss: number;
  validationLoss: number;
  trainAccuracy: number;
  validationAccuracy: number;
}

export interface BoundarySnapshot {
  epoch: number;
  grid: GridPoint[];
}

export interface PlaygroundPayload {
  samples: Sample[];
  grid: GridPoint[];
  snapshots: BoundarySnapshot[];
  history: HistoryRow[];
  epoch: number;
  targetEpochs: number;
}

export const DEFAULT_CONFIG: PlaygroundConfig = {
  dataset: "circles",
  sampleCount: 256,
  noise: 0.12,
  trainRatio: 0.75,
  hiddenLayers: [8, 8],
  activation: "tanh",
  optimizer: "Adam",
  learningRate: 0.03,
  epochs: 80,
  batchSize: 32,
  seed: 7,
  featureNames: [...DEFAULT_FEATURES],
  gridSize: 100,
};

export const PLAYGROUND_PRESETS: Record<string, PlaygroundConfig> = {
  circles: {
    ...DEFAULT_CONFIG,
    dataset: "circles",
    sampleCount: 320,
    noise: 0.08,
    hiddenLayers: [12, 12],
    learningRate: 0.035,
    epochs: 90,
    gridSize: 100,
    seed: 11,
  },
  xor: {
    ...DEFAULT_CONFIG,
    dataset: "xor",
    sampleCount: 352,
    noise: 0.06,
    hiddenLayers: [16, 8],
    activation: "relu",
    learningRate: 0.025,
    epochs: 120,
    featureNames: ["x1", "x2", "x1_x2", "x1_squared", "x2_squared"],
    gridSize: 100,
    seed: 19,
  },
  spiral: {
    ...DEFAULT_CONFIG,
    dataset: "spiral",
    sampleCount: 448,
    noise: 0.1,
    hiddenLayers: [24, 16, 8],
    learningRate: 0.02,
    epochs: 160,
    batchSize: 48,
    featureNames: ["x1", "x2", "sin_x1", "sin_x2", "x1_x2"],
    gridSize: 100,
    seed: 29,
  },
  gaussian: {
    ...DEFAULT_CONFIG,
    dataset: "gaussian",
    sampleCount: 256,
    noise: 0.14,
    hiddenLayers: [6],
    learningRate: 0.04,
    epochs: 60,
    gridSize: 100,
    seed: 5,
  },
};
