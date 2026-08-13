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
  DATASETS_2D_CLASSIFICATION,
  DATASETS_2D_REGRESSION,
  DEFAULT_DATASET_2D_CLASSIFICATION,
  DEFAULT_DATASET_2D_REGRESSION,
  isDataset2DClassificationId,
  isDataset2DRegressionId,
  NUM_SAMPLES,
  regressSinSin,
  shuffle,
} from "./dataset";
export type {
  DataGenerator,
  Dataset2DClassificationId,
  Dataset2DRegressionId,
  DatasetId,
  Example2D,
} from "./dataset";

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
  PALETTE_HIGH,
  PALETTE_LOW,
  PALETTE_MID,
  PALETTE_ZERO_WHITE,
  mixProbabilityColor,
  probabilityColorRgb,
  probabilityToRgba,
  valueToRgb,
  valueToRgbZeroWhite,
  weightColor,
  weightColorQuantized,
  weightColorZeroWhite,
  weightMagnitude,
  weightValueNormalized,
} from "./colors";

export { boundaryToGridPoints, computeBoundaries, examplesToSamples } from "./boundary";

export {
  DEFAULT_NETWORK_CONFIG,
  PlaygroundEngine,
  PLAYGROUND_OPTIMIZERS,
  NETWORK_ACTIVATIONS,
  NETWORK_REGULARIZATIONS,
  NETWORK_REGULARIZATION_RATES,
  WEIGHT_INITS,
} from "./engine";
export type {
  ArchitecturePresetId,
  GraphNodeKind,
  GraphPosition,
  HeatmapPresetId,
  LossHistoryPoint,
  PlaygroundOptimizerId,
  NetworkActivationId,
  NetworkAnyDatasetId,
  NetworkDataMode,
  NetworkPlaygroundConfig,
  NetworkProblemType,
  NetworkRegularizationId,
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
  mlpColumnXsFromCounts,
  mlpLayerGap,
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
  DEFAULT_HEATMAP_PRESET,
  DENSITY,
  HEATMAP_PRESETS,
  HEATMAP_PRESET_IDS,
  NODE_BOUNDARY_DENSITY,
  NODE_CURVE_DENSITY,
  PLAY_BOUNDARY_STRIDE,
  PLAY_DISPLAY_DENSITY,
  PLAY_NODE_BOUNDARY_DENSITY,
  X_DOMAIN,
  heatmapPreset,
  playBoundaryStride,
} from "./constants";
export type { HeatmapPreset } from "./constants";

// Train worker client + protocol types for the network engine.
// Re-exported here so network consumers depend on a single subpath.
export { NetworkTrainClient, canUseTrainWorkers } from "../workers";
export type {
  NetworkTrainSnapshot,
  NetworkCommandArgs,
  NetworkDataPoint,
  NetworkWorkerPayload,
  NetworkTrainClientOptions,
} from "../workers";
