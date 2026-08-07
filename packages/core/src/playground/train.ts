import { gridPoints, makeDataset } from "./datasets";
import { applyNormalization, featureMatrix, normalizeFeatures } from "./features";
import {
  accuracy,
  applyFlatGradMeans,
  createMLP,
  crossEntropyLoss,
  flattenWeights,
  loadWeights,
  predictProbabilities,
  trainBatch,
  type MLP,
} from "./mlp";
import { createRng, shuffleIndices } from "./rng";
import type {
  BoundarySnapshot,
  GridPoint,
  HistoryRow,
  PlaygroundConfig,
  PlaygroundPayload,
  Sample,
} from "./types";

export interface TrainingSplit {
  samples: Sample[];
  trainFeatures: number[][];
  trainLabels: number[];
  validationFeatures: number[][];
  validationLabels: number[];
  mean: number[];
  std: number[];
}

export interface LiveTrainingState {
  config: PlaygroundConfig;
  mlp: MLP;
  split: TrainingSplit;
  history: HistoryRow[];
  snapshots: Array<{ epoch: number; weights: Float64Array }>;
  epoch: number;
  playing: boolean;
}

function configSignature(config: PlaygroundConfig): string {
  return JSON.stringify({
    dataset: config.dataset,
    sampleCount: config.sampleCount,
    noise: config.noise,
    trainRatio: config.trainRatio,
    hiddenLayers: config.hiddenLayers,
    activation: config.activation,
    optimizer: config.optimizer,
    learningRate: config.learningRate,
    epochs: config.epochs,
    batchSize: config.batchSize,
    seed: config.seed,
    featureNames: config.featureNames,
    gridSize: config.gridSize,
  });
}

export function prepareTrainingSplit(config: PlaygroundConfig): TrainingSplit {
  const samples = makeDataset(config);
  const features = featureMatrix(samples, config.featureNames);
  const labels = samples.map((sample) => sample.target);
  const trainCount = Math.max(8, Math.floor(samples.length * config.trainRatio));
  const rng = createRng(config.seed + 17);
  const order = shuffleIndices(samples.length, rng);
  const trainIndices = order.slice(0, trainCount);
  const validationIndices = order.slice(trainCount);

  const trainFeaturesRaw = trainIndices.map((index) => features[index]);
  const { normalized, mean, std } = normalizeFeatures(trainFeaturesRaw);

  return {
    samples,
    trainFeatures: normalized,
    trainLabels: trainIndices.map((index) => labels[index]),
    validationFeatures: applyNormalization(
      validationIndices.map((index) => features[index]),
      mean,
      std,
    ),
    validationLabels: validationIndices.map((index) => labels[index]),
    mean,
    std,
  };
}

export function boundarySnapshotEpochs(epochs: number): Set<number> {
  const bounded = Math.max(1, epochs);
  const candidates = new Set<number>([0, bounded]);
  for (const fraction of [0.1, 0.25, 0.5, 0.75]) {
    candidates.add(Math.max(1, Math.round(bounded * fraction)));
  }
  return candidates;
}

function decisionGrid(
  mlp: MLP,
  config: PlaygroundConfig,
  mean: number[],
  std: number[],
): GridPoint[] {
  const points = gridPoints(config.gridSize);
  const features = applyNormalization(featureMatrix(points, config.featureNames), mean, std);
  const probabilities = predictProbabilities(mlp, features);
  return points.map((point, index) => ({
    x1: point.x1,
    x2: point.x2,
    probability: probabilities[index],
  }));
}

function appendHistoryRow(
  history: HistoryRow[],
  mlp: MLP,
  split: TrainingSplit,
  epoch: number,
): void {
  if (history.length && history[history.length - 1].epoch === epoch) return;
  history.push({
    epoch,
    trainLoss: crossEntropyLoss(mlp, split.trainFeatures, split.trainLabels),
    validationLoss: crossEntropyLoss(mlp, split.validationFeatures, split.validationLabels),
    trainAccuracy: accuracy(mlp, split.trainFeatures, split.trainLabels),
    validationAccuracy: accuracy(mlp, split.validationFeatures, split.validationLabels),
  });
}

export function createLiveTrainingState(config: PlaygroundConfig): LiveTrainingState {
  const split = prepareTrainingSplit(config);
  const inputSize = split.trainFeatures[0]?.length ?? config.featureNames.length;
  const mlp = createMLP(
    inputSize,
    config.hiddenLayers,
    config.activation,
    config.optimizer,
    config.learningRate,
    config.seed,
  );
  const history: HistoryRow[] = [];
  const snapshots: Array<{ epoch: number; weights: Float64Array }> = [
    { epoch: 0, weights: flattenWeights(mlp) },
  ];
  appendHistoryRow(history, mlp, split, 0);
  return { config, mlp, split, history, snapshots, epoch: 0, playing: false };
}

function trainOneEpoch(state: LiveTrainingState): void {
  const { config, mlp, split } = state;
  const batchSize = Math.max(4, Math.min(config.batchSize, split.trainFeatures.length));
  const order = shuffleIndices(split.trainFeatures.length, createRng(config.seed + state.epoch + 101));
  for (let start = 0; start < split.trainFeatures.length; start += batchSize) {
    const indices = order.slice(start, start + batchSize);
    const batch = indices.map((index) => split.trainFeatures[index]);
    const labels = indices.map((index) => split.trainLabels[index]);
    trainBatch(mlp, batch, labels);
  }
}

/**
 * One epoch with an external batch trainer (data-parallel coordinator).
 * `applyBatch(indices)` must update `state.mlp` weights for that mini-batch.
 */
export async function trainOneEpochWith(
  state: LiveTrainingState,
  applyBatch: (indices: number[]) => void | Promise<void>,
): Promise<void> {
  const { config, split } = state;
  const batchSize = Math.max(4, Math.min(config.batchSize, split.trainFeatures.length));
  const order = shuffleIndices(split.trainFeatures.length, createRng(config.seed + state.epoch + 101));
  for (let start = 0; start < split.trainFeatures.length; start += batchSize) {
    const indices = order.slice(start, start + batchSize);
    await applyBatch(indices);
  }
}

/** Apply flat grad sums (from shards) as a mean update on the live MLP. */
export function applyMlpGradSums(mlp: MLP, gradSums: Float64Array, count: number): void {
  const n = Math.max(1, count);
  const means = new Float64Array(gradSums.length);
  for (let i = 0; i < gradSums.length; i++) means[i] = gradSums[i]! / n;
  applyFlatGradMeans(mlp, means);
}

export function finishEpochBookkeeping(state: LiveTrainingState): LiveTrainingState {
  const next = { ...state, history: [...state.history], snapshots: [...state.snapshots] };
  next.epoch += 1;
  appendHistoryRow(next.history, next.mlp, next.split, next.epoch);
  const snapshotEpochs = boundarySnapshotEpochs(next.config.epochs);
  if (snapshotEpochs.has(next.epoch)) {
    next.snapshots.push({ epoch: next.epoch, weights: flattenWeights(next.mlp) });
  }
  if (next.epoch >= next.config.epochs) next.playing = false;
  return next;
}

export function advanceLiveTraining(state: LiveTrainingState, epochs = 1): LiveTrainingState {
  const next = { ...state, history: [...state.history], snapshots: [...state.snapshots] };
  const targetEpoch = Math.min(next.config.epochs, next.epoch + Math.max(1, epochs));
  const snapshotEpochs = boundarySnapshotEpochs(next.config.epochs);

  while (next.epoch < targetEpoch) {
    next.epoch += 1;
    trainOneEpoch(next);
    appendHistoryRow(next.history, next.mlp, next.split, next.epoch);
    if (snapshotEpochs.has(next.epoch)) {
      next.snapshots.push({ epoch: next.epoch, weights: flattenWeights(next.mlp) });
    }
  }

  if (next.epoch >= next.config.epochs) next.playing = false;
  return next;
}

export function resetLiveTraining(config: PlaygroundConfig): LiveTrainingState {
  return createLiveTrainingState(config);
}

export function buildPayload(state: LiveTrainingState): PlaygroundPayload {
  const { config, mlp, split, history, snapshots, epoch } = state;
  const saved = flattenWeights(mlp);
  const snapshotFrames: BoundarySnapshot[] = snapshots.map(({ epoch: snapshotEpoch, weights }) => {
    loadWeights(mlp, weights);
    return {
      epoch: snapshotEpoch,
      grid: decisionGrid(mlp, config, split.mean, split.std),
    };
  });
  loadWeights(mlp, saved);

  const currentGrid = decisionGrid(mlp, config, split.mean, split.std);
  return {
    samples: split.samples,
    grid: currentGrid,
    snapshots: snapshotFrames,
    history,
    epoch,
    targetEpochs: config.epochs,
  };
}

export function frameIndexForEpoch(snapshots: BoundarySnapshot[], epoch: number): number {
  if (!snapshots.length) return 0;
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let i = 0; i < snapshots.length; i++) {
    const distance = Math.abs(snapshots[i].epoch - epoch);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

export function gridForEpoch(state: LiveTrainingState, epoch: number): GridPoint[] {
  const snapshot = state.snapshots.find((item) => item.epoch === epoch);
  if (!snapshot) {
    return decisionGrid(state.mlp, state.config, state.split.mean, state.split.std);
  }
  const saved = flattenWeights(state.mlp);
  loadWeights(state.mlp, snapshot.weights);
  const grid = decisionGrid(state.mlp, state.config, state.split.mean, state.split.std);
  loadWeights(state.mlp, saved);
  return grid;
}

export { configSignature };
