export {
  Activations,
  backProp,
  buildNetwork,
  Errors,
  forwardProp,
  forEachNode,
  getOutputNode,
  Link,
  Node,
  RegularizationFunction,
  updateWeights,
} from "./nn";
export type { ActivationFunction, ErrorFunction, RegularizationFunction as RegularizationFn } from "./nn";

export {
  constructInput,
  constructInputIds,
  INPUT_IDS,
  INPUTS,
} from "./inputs";
export type { InputFeature } from "./inputs";

export {
  classifyCircleData,
  classifySpiralData,
  classifyTwoGaussData,
  classifyXORData,
  DATASETS,
  NUM_SAMPLES,
  shuffle,
} from "./dataset";
export type { DataGenerator, DatasetId, Example2D } from "./dataset";

export {
  DATASETS_1D,
  DATASETS_1D_CLASSIFICATION,
  DATASETS_1D_REGRESSION,
  DEFAULT_DATASET_1D_CLASSIFICATION,
  DEFAULT_DATASET_1D_REGRESSION,
  FEATURES_1D,
  FEATURES_2D_ONLY,
  isDataset1DClassificationId,
  isDataset1DId,
  isDataset1DRegressionId,
  targetCurve1D,
} from "./dataset-1d";
export type {
  Dataset1DClassificationId,
  Dataset1DId,
  Dataset1DRegressionId,
  Example1D,
} from "./dataset-1d";

export {
  CLASS_0_HEX,
  CLASS_1_HEX,
  mixProbabilityColor,
  probabilityColorRgb,
  probabilityToRgba,
  valueToRgb,
  weightColor,
  weightColorQuantized,
  weightMagnitude,
  weightValueNormalized,
} from "./colors";

export { boundaryToGridPoints, computeBoundaries, examplesToSamples } from "./boundary";

export {
  DEFAULT_TF_CONFIG,
  PlaygroundEngine,
  TF_ACTIVATIONS,
  TF_REGULARIZATIONS,
  TF_REGULARIZATION_RATES,
  WEIGHT_INITS,
} from "./engine";
export type {
  ArchitecturePresetId,
  GraphNodeKind,
  GraphPosition,
  LossHistoryPoint,
  TfActivationId,
  TfAnyDatasetId,
  TfDataMode,
  TfPlaygroundConfig,
  TfProblemType,
  TfRegularizationId,
  WeightInitId,
} from "./engine";

export {
  applyArchitecturePreset,
  backPropGraph,
  buildMlpGraph,
  buildMiniResNetGraph,
  buildResBlockGraph,
  ComputationalGraph,
  computeGraphBoundaries,
  forwardPropGraph,
  forEachGraphNode,
  GraphNode,
  initGraphCurveStore,
  layoutMlpFromLayers,
  MLP_COL_SPACING,
  MLP_NODE_SIZE,
  MLP_OUTPUT_NODE_SIZE,
  normalizeGraphLayout,
  PALETTE_NODE_KINDS,
  presetLabel,
  updateGraphCurves,
  updateWeightsGraph,
} from "./graph";
export type {
  GraphEdgeDef,
  GraphSnapshot,
  PaletteNodeKind,
} from "./graph";

export {
  CURVE_DENSITY,
  DENSITY,
  NODE_BOUNDARY_DENSITY,
  NODE_CURVE_DENSITY,
  PLAY_DISPLAY_DENSITY,
  X_DOMAIN,
} from "./constants";
