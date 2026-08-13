/**
 * Shared message protocol for playground train workers.
 * Worker owns the engine; main thread paints from snapshots.
 */

import type { CnnConfig, FeatureMapSnapshot, TrainingStats } from "../cnn/engine";
import type { LayerKind, LayerShape } from "../cnn/layers/base";
import type { ImageExample, SignalExample } from "../cnn/gallery";
import type { LossHistoryPoint, NetworkPlaygroundConfig } from "../network/engine";
import type { PlaygroundConfig, PlaygroundPayload } from "../types";

export type TrainRebuildReason = "topology" | "dataset" | "reset" | "resetWeights" | "mode";

/** Serializable CNN layer row for React Flow (no live Layer instances). */
export interface CnnLayerView {
  id: string;
  kind: LayerKind;
  label: string;
  shape: LayerShape;
  params: number;
  weightMag: number | null;
}

export interface CnnTrainSnapshot {
  kind: "cnn";
  config: CnnConfig;
  stats: TrainingStats;
  lossHistory: LossHistoryPoint[];
  layers: CnnLayerView[];
  featureMaps: FeatureMapSnapshot[];
  kernels: Record<string, number[][] | number[][][]>;
  galleryExamples: Array<ImageExample | SignalExample>;
  galleryPredictions: number[];
  inspectedExampleIndex: number;
  loss: number;
  probability: number;
}

export interface NetworkTrainSnapshot {
  kind: "network";
  config: NetworkPlaygroundConfig;
  epoch: number;
  lossTrain: number;
  lossTest: number;
  lossHistory: LossHistoryPoint[];
  /** Node id → boundary heatmap matrix. */
  boundary: Record<string, number[][]>;
  /** Node id → activation curve samples. */
  curves: Record<string, number[]>;
  targetCurve: number[] | null;
  /** Full graph weights/topology for syncing the main-thread display engine. */
  graphSnapshot: import("../network/graph/types").GraphSnapshot;
  /**
   * Train/test data — only sent on full snapshots (ready/rebuilt/pause/step),
   * NOT on high-frequency ticks. The main-thread engine is the source of truth
   * for data; cloning these every paint tick (≤30 Hz) was pure waste.
   */
  trainData?: Array<{ x: number; y: number; label: number }>;
  testData?: Array<{ x: number; y: number; label: number }>;
}

export interface MlpTrainSnapshot {
  kind: "mlp";
  config: PlaygroundConfig;
  epoch: number;
  playing: boolean;
  payload: PlaygroundPayload;
}

export type TrainSnapshot = CnnTrainSnapshot | NetworkTrainSnapshot | MlpTrainSnapshot;

export type ToTrainWorker =
  | { type: "init"; config: unknown }
  | { type: "setConfig"; patch: Record<string, unknown> }
  | { type: "rebuild"; reason: TrainRebuildReason; payload?: unknown }
  | { type: "play"; epochsPerSec: number }
  | { type: "pause" }
  | { type: "step" }
  | { type: "inspect"; exampleIndex?: number }
  | { type: "command"; name: string; args?: unknown }
  | { type: "dispose" };

export type FromTrainWorker =
  | { type: "ready"; snapshot: TrainSnapshot }
  | { type: "tick"; snapshot: TrainSnapshot }
  | { type: "rebuilt"; reason: TrainRebuildReason; snapshot: TrainSnapshot }
  | { type: "error"; message: string };
