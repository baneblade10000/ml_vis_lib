/**
 * Legacy 2D MLP decision-boundary engine — the `@ml-vis/core/mlp` subpath.
 *
 * CNN / network / autograd each have their own subpath; this barrel is mlp-only,
 * which lets it export `DATASETS` / `DatasetId` / `OPTIMIZERS` without colliding
 * with the network engine's same-named exports (the collision that previously
 * forced the root barrel to alias them as `NETWORK_DATASETS` etc.).
 */
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

// Train worker client + protocol types for the MLP engine. Re-exported here so
// MLP consumers depend on a single subpath.
export { MlpTrainClient, canUseTrainWorkers } from "./workers";
export type { MlpTrainSnapshot, MlpTrainClientOptions } from "./workers";
