export * from "./types";
export { makeDataset, gridPoints } from "./datasets";
export { featureMatrix, normalizeFeatures } from "./features";
export { createMLP, predictProbabilities, accuracy } from "./mlp";
export {
  advanceLiveTraining,
  buildPayload,
  createLiveTrainingState,
  frameIndexForEpoch,
  gridForEpoch,
  prepareTrainingSplit,
  resetLiveTraining,
  configSignature,
} from "./train";
export type { LiveTrainingState, TrainingSplit } from "./train";

// Convolutional-network engine (isolated from the feedforward `tf` engine).
export {
  CnnEngine,
  DEFAULT_CNN_CONFIG,
  DEFAULT_CNN_CONFIG_2D,
  DEFAULT_CNN_CONFIG_1D,
  CNN_DATASET_IDS_2D,
  CNN_DATASET_IDS_1D,
  CNN_ACTIVATION_IDS,
  CNN_REGULARIZATIONS,
  CNN_REGULARIZATION_RATES,
  PLAYGROUND_OPTIMIZERS,
  IMAGE_SIZE,
  SIGNAL_LENGTH,
} from "./cnn";
export type {
  CnnMode,
  LayerSpec,
  CnnConfig,
  TrainingStats as CnnTrainingStats,
  FeatureMapSnapshot,
  ImageExample,
  SignalExample,
  CnnDatasetId,
  CnnDatasetId2D,
  CnnDatasetId1D,
  CnnActivationId,
  CnnRegularizationId,
  PlaygroundOptimizerId,
  LayerKind,
  LayerShape,
  Map2D,
  Volume,
  Signal as CnnSignal,
} from "./cnn";
