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
} from "./engine";
export type {
  ArchitecturePresetId,
  GraphNodeKind,
  GraphPosition,
  LossHistoryPoint,
  TfActivationId,
  TfPlaygroundConfig,
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
  layoutMlpFromLayers,
  MLP_COL_SPACING,
  MLP_NODE_SIZE,
  MLP_OUTPUT_NODE_SIZE,
  normalizeGraphLayout,
  PALETTE_NODE_KINDS,
  presetLabel,
  updateWeightsGraph,
} from "./graph";
export type {
  GraphEdgeDef,
  GraphSnapshot,
  PaletteNodeKind,
} from "./graph";

export { DENSITY, NODE_BOUNDARY_DENSITY, PLAY_DISPLAY_DENSITY, X_DOMAIN } from "./constants";
