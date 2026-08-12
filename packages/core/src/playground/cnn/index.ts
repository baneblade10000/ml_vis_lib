export {
  IMAGE_SIZE,
  IMAGE_SIZE_THREE_FOUR_LOOPS,
  imageSizeForDataset,
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

export { CNN_ACTIVATION_IDS } from "./activations";
export type { CnnActivationId } from "./activations";

export { zeros2D } from "./tensor";
export type { Map2D, Volume, Signal } from "./tensor";

export type { LayerKind, DataSpace, LayerShape } from "./layers/base";
export { shapeSpace } from "./layers/base";

export {
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
  PoolKind,
} from "./engine";
