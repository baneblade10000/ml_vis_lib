export {
  IMAGE_SIZE,
  SIGNAL_LENGTH,
  NUM_EXAMPLES,
  DEFAULT_TRAIN_RATIO,
  CNN_DATASET_IDS_2D,
  CNN_DATASET_IDS_1D,
  makeImageDataset,
  makeSignalDataset,
  imageToVolume,
  signalToInput,
} from "./gallery";
export type {
  ImageExample,
  SignalExample,
  CnnDatasetId,
  CnnDatasetId2D,
  CnnDatasetId1D,
} from "./gallery";

export { CNN_ACTIVATIONS, CNN_ACTIVATION_IDS, activationById, Activations } from "./activations";
export type { CnnActivationId } from "./activations";
export type { ActivationFunction } from "../tf/nn";

export { Losses } from "./loss";
export type { LossFunction } from "./loss";

export {
  zeros2D,
  zerosVolume,
  zeros1D,
  zerosSignal,
  clone2D,
  cloneVolume,
  clone1D,
  cloneSignal,
  cloneKernel2D,
  cloneKernel1D,
  map2DInPlace,
  sum2D,
  sumVolume,
  sum1D,
  gaussian,
  randomInit,
} from "./tensor";
export type {
  Map2D,
  Volume,
  Signal,
  Kernel2D,
  Kernel1D,
  Spatial,
  VolumeShape,
  SignalShape,
} from "./tensor";

export { Layer } from "./layers/base";
export type { LayerKind, DataSpace, LayerShape } from "./layers/base";
export { flattenVolume, unflattenVolume } from "./layers/base";
export { InputLayer } from "./layers/input";
export { Conv2DLayer } from "./layers/conv2d";
export { Pool2DLayer } from "./layers/pool2d";
export type { PoolKind2D } from "./layers/pool2d";
export { Conv1DLayer } from "./layers/conv1d";
export { Pool1DLayer } from "./layers/pool1d";
export { FlattenLayer } from "./layers/flatten";
export { DenseLayer } from "./layers/dense";
export { OutputLayer } from "./layers/output";

export {
  CnnEngine,
  DEFAULT_CNN_CONFIG,
  DEFAULT_CNN_CONFIG_2D,
  DEFAULT_CNN_CONFIG_1D,
  CNN_REGULARIZATIONS,
  CNN_REGULARIZATION_RATES,
  PLAYGROUND_OPTIMIZERS,
} from "./engine";
export type {
  CnnMode,
  LayerSpec,
  CnnConfig,
  TrainingStats,
  FeatureMapSnapshot,
  CnnRegularizationId,
  PlaygroundOptimizerId,
} from "./engine";
export { applyRegularizedUpdate, regularizationDerivative } from "./regularization";
