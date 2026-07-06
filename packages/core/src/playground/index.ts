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
