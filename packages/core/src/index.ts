export {
  computeSectionLayout,
  SectionRegistry,
  SectionStore,
  sortSections,
} from "./section";
export type {
  SectionDefinition,
  SectionLayoutOptions,
  SectionLayoutResult,
  SectionRect,
  SectionSize,
  SectionState,
  SectionStoreSnapshot,
} from "./section";

export { downsample, extent } from "./utils/math";

export {
  DEFAULT_MONTE_CARLO_CONFIG,
  MonteCarloPiEngine,
  PI_TRUE,
  estimateError,
  estimatePi,
  isInsideUnitQuarterCircle,
} from "./montecarlo";
export type {
  MonteCarloConfig,
  MonteCarloHistoryPoint,
  MonteCarloPoint,
} from "./montecarlo";

export {
  ACTIVATIONS,
  DATASETS,
  DEFAULT_CONFIG,
  OPTIMIZERS,
  PLAYGROUND_PRESETS,
  advanceLiveTraining,
  buildPayload,
  createLiveTrainingState,
  createMLP,
  frameIndexForEpoch,
  gridForEpoch,
  makeDataset,
  resetLiveTraining,
  configSignature,
} from "./playground";
export type {
  ActivationId,
  BoundarySnapshot,
  DatasetId,
  FeatureName,
  GridPoint,
  HistoryRow,
  LiveTrainingState,
  OptimizerId,
  PlaygroundConfig,
  PlaygroundPayload,
  Sample,
} from "./playground";

export { DecisionBoundaryPlot } from "./charts/decision-boundary-plot";
export type { DecisionBoundaryPayload } from "./charts/decision-boundary-plot";

export { reduceMatrix, renderValueMatrix } from "./charts/mini-heatmap";

export {
  Activations,
  backProp,
  boundaryToGridPoints,
  buildNetwork,
  classifyCircleData,
  classifySpiralData,
  classifyTwoGaussData,
  classifyXORData,
  CLASS_0_HEX,
  CLASS_1_HEX,
  computeBoundaries,
  constructInput,
  constructInputIds,
  DATASETS as TF_DATASETS,
  DEFAULT_TF_CONFIG,
  DENSITY,
  NODE_BOUNDARY_DENSITY,
  PLAY_DISPLAY_DENSITY,
  Errors,
  examplesToSamples,
  forwardProp,
  forEachNode,
  getOutputNode,
  INPUT_IDS,
  INPUTS,
  Link,
  Node,
  NUM_SAMPLES,
  PlaygroundEngine,
  RegularizationFunction,
  shuffle,
  TF_ACTIVATIONS,
  updateWeights,
  valueToRgb,
  weightColor,
  X_DOMAIN,
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
  PALETTE_NODE_KINDS,
  presetLabel,
  updateWeightsGraph,
} from "./playground/tf";
export type {
  ArchitecturePresetId,
  DataGenerator as TfDataGenerator,
  DatasetId as TfDatasetId,
  Example2D,
  GraphEdgeDef,
  GraphNodeKind,
  GraphPosition,
  GraphSnapshot,
  InputFeature,
  LossHistoryPoint,
  PaletteNodeKind,
  TfActivationId,
  TfPlaygroundConfig,
} from "./playground/tf";

export {
  AutogradGraph,
  AutogradNode,
  AutogradEdge,
  AUTOGRAD_PALETTE_OPS,
  AUTOGRAD_PRESETS,
  buildAutogradPreset,
  CompGraphEngine,
  DEFAULT_COMPGRAPH_CONFIG,
  evaluateOp,
  localDerivatives,
  OP_SPECS,
  resetAutogradIdCounter,
} from "./playground/autograd";
export type {
  AutogradEdgeDef,
  AutogradNodeDef,
  AutogradOp,
  AutogradPosition,
  AutogradPresetId,
  AutogradSnapshot,
  CompGraphConfig,
  OpSpec,
} from "./playground/autograd";

export {
  coreMessages,
  getLocale,
  localeLabels,
  locales,
  setLocale,
  subscribeLocale,
  t,
} from "./i18n";
export type { CoreMessages, Locale } from "./i18n";
