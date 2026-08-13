/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {
  DATASETS,
  DEFAULT_DATASET_2D_CLASSIFICATION,
  DEFAULT_DATASET_2D_REGRESSION,
  NUM_SAMPLES,
  shuffle,
  type DataGenerator,
  type DatasetId,
  type Example2D,
} from "./dataset";
import {
  DATASETS_1D,
  DEFAULT_DATASET_1D_CLASSIFICATION,
  DEFAULT_DATASET_1D_REGRESSION,
  FEATURES_2D_ONLY,
  isDataset1DId,
  targetCurve1D,
  type Dataset1DId,
} from "./dataset-1d";
import { computeBoundaries } from "./boundary";
import { constructInput, constructInputIds } from "./inputs";
import {
  Activations,
  Errors,
  RegularizationFunction,
  type ActivationFunction,
  type Node,
  type RegularizationFunction as RegFn,
} from "./nn";
import {
  applyArchitecturePreset,
  backPropGraph,
  buildMlpGraph,
  ComputationalGraph,
  forwardPropGraph,
  GraphNode,
  initGraphBoundaryStore,
  initGraphCurveStore,
  updateGraphBoundaries,
  updateGraphCurves,
  updateGraphHiddenBoundaries,
  updateGraphHiddenCurves,
  updateGraphInputFeatures,
  updateGraphOutputBoundary,
  updateGraphOutputCurve,
  updateWeightsGraph,
  type ArchitecturePresetId,
  type GraphNodeKind,
  type GraphPosition,
} from "./graph";
import { normalizeGraphLayout } from "./graph/mlp-layout";
import {
  CURVE_DENSITY,
  DEFAULT_HEATMAP_PRESET,
  PLAY_CURVE_STRIDE,
  heatmapPreset,
  playBoundaryStride,
  type HeatmapPreset,
  type HeatmapPresetId,
} from "./constants";
import { sampleBias, sampleWeight, type WeightInitId } from "./weight-init";

import type { PlaygroundOptimizerId } from "../optimizers";
export type { PlaygroundOptimizerId } from "../optimizers";
export { PLAYGROUND_OPTIMIZERS } from "../optimizers";

export type NetworkActivationId = "relu" | "tanh" | "sigmoid" | "linear";
export type NetworkRegularizationId = "none" | "L1" | "L2";
export type NetworkDataMode = "2d" | "1d";
export type NetworkProblemType = "classification" | "regression";
export type NetworkAnyDatasetId = DatasetId | Dataset1DId;
export type { WeightInitId } from "./weight-init";
export type { Dataset1DId } from "./dataset-1d";
export { WEIGHT_INITS } from "./weight-init";

export const NETWORK_ACTIVATIONS: Record<NetworkActivationId, ActivationFunction> = {
  relu: Activations.RELU,
  tanh: Activations.TANH,
  sigmoid: Activations.SIGMOID,
  linear: Activations.LINEAR,
};

export const NETWORK_REGULARIZATIONS: readonly NetworkRegularizationId[] = ["none", "L1", "L2"];

export const NETWORK_REGULARIZATION_RATES = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10] as const;

const DEFAULT_FEATURES_2D: Record<string, boolean> = {
  x: true,
  y: true,
  xSquared: false,
  ySquared: false,
  xTimesY: false,
  sinX: false,
  sinY: false,
};

const DEFAULT_FEATURES_1D: Record<string, boolean> = {
  x: true,
  y: false,
  xSquared: false,
  ySquared: false,
  xTimesY: false,
  sinX: false,
  sinY: false,
};

export interface NetworkPlaygroundConfig {
  learningRate: number;
  optimizer: PlaygroundOptimizerId;
  activation: NetworkActivationId;
  weightInit: WeightInitId;
  regularization: NetworkRegularizationId;
  regularizationRate: number;
  batchSize: number;
  noise: number;
  percTrainData: number;
  dataMode: NetworkDataMode;
  problemType: NetworkProblemType;
  dataset: NetworkAnyDatasetId;
  networkShape: number[];
  numHiddenLayers: number;
  enabledFeatures: Record<string, boolean>;
  discretize?: boolean;
  architecturePreset: ArchitecturePresetId;
  heatmapPreset: HeatmapPresetId;
}

export const DEFAULT_NETWORK_CONFIG: NetworkPlaygroundConfig = {
  learningRate: 0.03,
  optimizer: "SGD",
  activation: "tanh",
  weightInit: "uniform",
  regularization: "none",
  regularizationRate: 0,
  batchSize: 10,
  noise: 0,
  percTrainData: 50,
  dataMode: "2d",
  problemType: "classification",
  dataset: "circle",
  networkShape: [4],
  numHiddenLayers: 1,
  enabledFeatures: { ...DEFAULT_FEATURES_2D },
  discretize: false,
  architecturePreset: "mlp",
  heatmapPreset: DEFAULT_HEATMAP_PRESET,
};

export interface LossHistoryPoint {
  epoch: number;
  train: number;
  test: number;
}

/** Cap so long Play sessions don't grow unbounded (~1 min at ~20 Hz sample rate). */
const LOSS_HISTORY_MAX = 1200;

export class PlaygroundEngine {
  config: NetworkPlaygroundConfig;
  graph: ComputationalGraph = new ComputationalGraph();
  boundary: Record<string, number[][]> = {};
  /** 1D activation curves per node (used when dataMode === "1d"). */
  curves: Record<string, number[]> = {};
  /** Ideal regression target curve, if any. */
  targetCurve: number[] | null = null;
  trainData: Example2D[] = [];
  testData: Example2D[] = [];
  lossTrain = 0;
  lossTest = 0;
  epoch = 0;
  lossHistory: LossHistoryPoint[] = [];
  /** 1-based Adam/RMSProp step; reset on weight reinit or optimizer change. */
  optStep = 0;
  private boundaryNeedsInputRefresh = true;
  /** Config snapshot from first page load — used by resetToInitial(). */
  private readonly initialConfig: NetworkPlaygroundConfig;

  constructor(config: Partial<NetworkPlaygroundConfig> = {}) {
    this.config = this.cloneConfig(this.mergeConfig(config));
    this.initialConfig = this.cloneConfig(this.config);
    this.bootstrap();
  }

  private mergeConfig(config: Partial<NetworkPlaygroundConfig>): NetworkPlaygroundConfig {
    return {
      ...DEFAULT_NETWORK_CONFIG,
      ...config,
      networkShape: [...(config.networkShape ?? DEFAULT_NETWORK_CONFIG.networkShape)],
      enabledFeatures: { ...(config.enabledFeatures ?? DEFAULT_NETWORK_CONFIG.enabledFeatures) },
    };
  }

  private cloneConfig(config: NetworkPlaygroundConfig): NetworkPlaygroundConfig {
    return {
      ...config,
      networkShape: [...config.networkShape],
      enabledFeatures: { ...config.enabledFeatures },
    };
  }

  /** First-time / factory setup from current config. */
  private bootstrap(): void {
    this.epoch = 0;
    this.lossHistory = [];
    this.generateData();
    this.rebuildNetwork();
    this.lossTrain = this.getLoss(this.trainData);
    this.lossTest = this.getLoss(this.testData);
    this.boundaryNeedsInputRefresh = true;
    this.rebuildBoundaryStore();
    this.refreshBoundary();
    this.lossHistory.push({ epoch: 0, train: this.lossTrain, test: this.lossTest });
  }

  /** Layered view for backward-compatible tooling. */
  get network(): Node[][] {
    return this.graph.toLayeredNetwork() as Node[][];
  }

  reset(): void {
    this.epoch = 0;
    this.lossHistory = [];
    this.generateData();
    // Soft reset: keep a hand-built topology, only reinit weights; rebuild presets.
    if (this.config.architecturePreset === "custom" && this.graph.nodes.size > 0) {
      this.reinitializeWeights();
    } else {
      this.rebuildNetwork();
    }
    this.lossTrain = this.getLoss(this.trainData);
    this.lossTest = this.getLoss(this.testData);
    this.boundaryNeedsInputRefresh = true;
    this.rebuildBoundaryStore();
    this.refreshBoundary();
    this.lossHistory.push({ epoch: 0, train: this.lossTrain, test: this.lossTest });
  }

  /**
   * Restore architecture/hyperparameters from first page load, but keep the
   * active data layout (2D/1D, problem type, dataset, features).
   */
  resetToInitial(): void {
    const dataMode = this.config.dataMode;
    const problemType = this.config.problemType;
    const dataset = this.config.dataset;
    const enabledFeatures = { ...this.config.enabledFeatures };
    const heatmapPresetId = this.config.heatmapPreset;

    this.config = this.cloneConfig(this.initialConfig);
    this.config.heatmapPreset = heatmapPresetId;
    this.config.dataMode = dataMode;

    if (dataMode === "1d") {
      this.config.problemType = problemType;
      this.config.dataset = isDataset1DId(dataset)
        ? dataset
        : problemType === "regression"
          ? DEFAULT_DATASET_1D_REGRESSION
          : DEFAULT_DATASET_1D_CLASSIFICATION;
      this.config.enabledFeatures = {
        ...DEFAULT_FEATURES_1D,
        x: !!enabledFeatures.x,
        xSquared: !!enabledFeatures.xSquared,
        sinX: !!enabledFeatures.sinX,
      };
      if (
        !this.config.enabledFeatures.x
        && !this.config.enabledFeatures.xSquared
        && !this.config.enabledFeatures.sinX
      ) {
        this.config.enabledFeatures.x = true;
      }
    } else {
      this.config.problemType = problemType;
      this.config.dataset =
        dataset in DATASETS
          ? (dataset as DatasetId)
          : problemType === "regression"
            ? DEFAULT_DATASET_2D_REGRESSION
            : DEFAULT_DATASET_2D_CLASSIFICATION;
      this.config.enabledFeatures =
        dataset in DATASETS
          ? { ...DEFAULT_FEATURES_2D, ...enabledFeatures }
          : { ...DEFAULT_FEATURES_2D };
      if (!Object.values(this.config.enabledFeatures).some(Boolean)) {
        this.config.enabledFeatures = { ...DEFAULT_FEATURES_2D };
      }
    }

    this.bootstrap();
  }

  /** Randomize weights/biases in place; topology and hyperparameters unchanged. */
  resetWeights(): void {
    this.epoch = 0;
    this.lossHistory = [];
    this.reinitializeWeights();
    this.lossTrain = this.getLoss(this.trainData);
    this.lossTest = this.getLoss(this.testData);
    this.boundaryNeedsInputRefresh = true;
    this.rebuildBoundaryStore();
    this.refreshBoundary();
    this.lossHistory.push({ epoch: 0, train: this.lossTrain, test: this.lossTest });
  }

  /** Fresh weights / biases in place using the selected init scheme. */
  private reinitializeWeights(): void {
    this.optStep = 0;
    const init = this.config.weightInit;
    for (const node of this.graph.nodes.values()) {
      if (node.kind === "input") continue;
      node.bias = sampleBias(init, node.kind);
      node.m = 0;
      node.v = 0;
      const fanIn = Math.max(node.inputLinks.length, 1);
      const fanOut = Math.max(node.outputs.length, 1);
      for (const link of node.inputLinks) {
        // Residual skip edges into Add nodes are identity (weight = 1).
        if (node.kind === "sum" && link.weight === 1) {
          link.isDead = false;
          link.m = 0;
          link.v = 0;
          continue;
        }
        link.weight = sampleWeight(init, fanIn, fanOut);
        link.isDead = false;
        link.errorDer = 0;
        link.accErrorDer = 0;
        link.numAccumulatedDers = 0;
        link.lastGradient = 0;
        link.m = 0;
        link.v = 0;
      }
    }
  }

  step(): void {
    this.trainEpoch();
    this.refreshMetrics();
    this.refreshBoundary();
    this.pushLossHistory();
  }

  /**
   * Apply weights + viz buffers from a train-worker tick onto this display
   * engine (topology must already match).
   */
  applyWorkerViz(payload: {
    epoch: number;
    lossTrain: number;
    lossTest: number;
    lossHistory: LossHistoryPoint[];
    boundary: Record<string, number[][]>;
    curves: Record<string, number[]>;
    targetCurve: number[] | null;
    graphSnapshot: import("./graph/types").GraphSnapshot;
    trainData?: Example2D[];
    testData?: Example2D[];
  }): void {
    this.epoch = payload.epoch;
    this.lossTrain = payload.lossTrain;
    this.lossTest = payload.lossTest;
    this.lossHistory = payload.lossHistory.map((p) => ({ ...p }));
    // Only adopt viz buffers when they cover the live output node — a stale
    // worker tick after addNeuron would otherwise blank the heatmaps.
    if (payload.boundary[this.graph.outputId]) {
      this.boundary = payload.boundary;
    }
    this.curves = payload.curves;
    this.targetCurve = payload.targetCurve;
    this.graph.applyWeightsFromSnapshot(payload.graphSnapshot);
    if (payload.trainData?.length) {
      this.trainData = payload.trainData.map((p) => ({ ...p }));
    }
    if (payload.testData?.length) {
      this.testData = payload.testData.map((p) => ({ ...p }));
    }
  }

  /**
   * Replace topology from a worker/main snapshot and rebuild viz buffers so
   * boundary keys match the new node ids (critical after freeform edits).
   */
  applyTopologySnapshot(
    snapshot: import("./graph/types").GraphSnapshot,
    options?: {
      trainData?: Example2D[];
      testData?: Example2D[];
      /** Reset epoch / loss history (default true — topology edits start fresh). */
      resetTraining?: boolean;
    },
  ): void {
    const reg = this.graph.regularization;
    this.graph = ComputationalGraph.fromSnapshot(snapshot, reg);
    this.syncOutputActivation();
    if (options?.trainData?.length) {
      this.trainData = options.trainData.map((p) => ({ ...p }));
    }
    if (options?.testData?.length) {
      this.testData = options.testData.map((p) => ({ ...p }));
    }
    if (options?.resetTraining !== false) {
      this.epoch = 0;
      this.optStep = 0;
      this.lossHistory = [];
    }
    this.boundaryNeedsInputRefresh = true;
    this.rebuildBoundaryStore();
    this.refreshMetrics();
    this.refreshBoundary();
    if (options?.resetTraining !== false) {
      this.lossHistory.push({ epoch: 0, train: this.lossTrain, test: this.lossTest });
    }
  }

  /**
   * Snapshot current train/test loss for the learning-curve chart.
   * Updates the last point in place when the epoch hasn't advanced (Play
   * refreshes metrics between epochs).
   */
  pushLossHistory(): void {
    const last = this.lossHistory[this.lossHistory.length - 1];
    if (last && last.epoch === this.epoch) {
      last.train = this.lossTrain;
      last.test = this.lossTest;
      return;
    }
    this.lossHistory.push({
      epoch: this.epoch,
      train: this.lossTrain,
      test: this.lossTest,
    });
    if (this.lossHistory.length > LOSS_HISTORY_MAX) {
      this.lossHistory.splice(0, this.lossHistory.length - LOSS_HISTORY_MAX);
    }
  }

  /** One training pass without boundary / test-loss refresh (for fast Play loop). */
  trainEpoch(updateTrainLoss = false): void {
    this.epoch++;
    const { batchSize, learningRate, regularizationRate, optimizer } = this.config;
    const yCoord = this.config.dataMode === "1d" ? 0 : undefined;
    this.trainData.forEach((point, i) => {
      const input = constructInput(
        point.x,
        yCoord === undefined ? point.y : yCoord,
        this.config.enabledFeatures,
      );
      forwardPropGraph(this.graph, input, i === 0);
      backPropGraph(this.graph, point.label, Errors.SQUARE);
      if ((i + 1) % batchSize === 0) {
        this.optStep += 1;
        updateWeightsGraph(this.graph, learningRate, regularizationRate, optimizer, this.optStep);
      }
    });
    if (updateTrainLoss) {
      this.lossTrain = this.getLoss(this.trainData);
    }
  }

  flattenParams(): Float64Array {
    return this.graph.flattenParams();
  }

  loadParams(vector: Float64Array): void {
    this.graph.loadParams(vector);
  }

  /**
   * Accumulate grad sums for the given trainData indices.
   * Caller must zeroGradAccumulators first. Returns examples processed.
   */
  accumulateGradIndices(indices: number[]): number {
    const yCoord = this.config.dataMode === "1d" ? 0 : undefined;
    let count = 0;
    for (let k = 0; k < indices.length; k++) {
      const point = this.trainData[indices[k]!];
      if (!point) continue;
      const input = constructInput(
        point.x,
        yCoord === undefined ? point.y : yCoord,
        this.config.enabledFeatures,
      );
      // Reset node ders on first sample only; keep acc* across the batch.
      forwardPropGraph(this.graph, input, k === 0);
      backPropGraph(this.graph, point.label, Errors.SQUARE);
      count++;
    }
    return count;
  }

  exportGradSums(): Float64Array {
    return this.graph.exportGradSums();
  }

  applyGradSums(vector: Float64Array, count: number): void {
    this.graph.loadGradSums(vector, count);
    const { learningRate, regularizationRate, optimizer } = this.config;
    this.optStep += 1;
    updateWeightsGraph(this.graph, learningRate, regularizationRate, optimizer, this.optStep);
  }

  zeroGradAccumulators(): void {
    this.graph.zeroGradAccumulators();
  }

  refreshMetrics(): void {
    this.lossTrain = this.getLoss(this.trainData);
    this.lossTest = this.getLoss(this.testData);
  }

  /** Full-quality boundary (2D) or curves (1D) for all nodes. */
  refreshBoundary(): void {
    if (this.config.dataMode === "1d") {
      this.refreshCurvesInternal(this.boundaryNeedsInputRefresh);
      return;
    }
    this.refreshBoundaryInternal(this.boundaryNeedsInputRefresh);
  }

  /**
   * Play-quality fill of the paused-size stores. Cheap enough to run on the
   * UI thread when adding layers/neurons so the graph can paint immediately;
   * a later {@link refreshBoundary} (worker or idle) restores unique cells.
   */
  refreshBoundaryPreview(): void {
    if (this.config.dataMode === "1d") {
      this.refreshCurvesInternal(this.boundaryNeedsInputRefresh);
      return;
    }
    const validation = this.graph.validate();
    if (!validation.valid) {
      this.boundary = computeBoundaries(this.network, this.config.enabledFeatures);
      this.boundaryNeedsInputRefresh = false;
      return;
    }
    const preset = this.heatmap();
    if (this.boundaryNeedsInputRefresh) {
      updateGraphInputFeatures(
        this.config.enabledFeatures,
        this.boundary,
        preset.hidden,
      );
    }
    this.refreshOutputBoundaryFast();
    this.refreshHiddenBoundariesFast();
    this.boundaryNeedsInputRefresh = false;
  }

  /** Fast output-only boundary/curve during Play (stride from heatmap preset). */
  refreshOutputBoundaryFast(stride?: number): void {
    const validation = this.graph.validate();
    if (!validation.valid) return;
    const preset = this.heatmap();
    const step = stride ?? playBoundaryStride(preset);
    if (this.config.dataMode === "1d") {
      updateGraphOutputCurve(
        this.graph,
        this.config.enabledFeatures,
        this.curves,
        this.graph.outputId,
        PLAY_CURVE_STRIDE,
      );
      return;
    }
    updateGraphOutputBoundary(
      this.graph,
      this.config.enabledFeatures,
      this.boundary,
      this.graph.outputId,
      step,
    );
  }

  /** Fast hidden-node heatmap/curve refresh during Play. */
  refreshHiddenBoundariesFast(): void {
    const validation = this.graph.validate();
    if (!validation.valid) return;
    if (this.config.dataMode === "1d") {
      updateGraphHiddenCurves(
        this.graph,
        this.config.enabledFeatures,
        this.curves,
        this.graph.outputId,
      );
      return;
    }
    const preset = this.heatmap();
    updateGraphHiddenBoundaries(
      this.graph,
      this.config.enabledFeatures,
      this.boundary,
      this.graph.outputId,
      preset.playHidden,
    );
  }

  heatmap(): HeatmapPreset {
    return heatmapPreset(this.config.heatmapPreset);
  }

  setHeatmapPreset(id: HeatmapPresetId): void {
    if (this.config.heatmapPreset === id) return;
    this.config.heatmapPreset = id;
    this.boundaryNeedsInputRefresh = true;
    this.rebuildBoundaryStore();
    this.refreshBoundary();
  }

  getLoss(dataPoints: Example2D[] = this.trainData): number {
    if (!dataPoints.length) return 0;
    let loss = 0;
    const yCoord = this.config.dataMode === "1d" ? 0 : undefined;
    for (const dataPoint of dataPoints) {
      const input = constructInput(
        dataPoint.x,
        yCoord === undefined ? dataPoint.y : yCoord,
        this.config.enabledFeatures,
      );
      const output = forwardPropGraph(this.graph, input);
      loss += Errors.SQUARE.error(output, dataPoint.label);
    }
    return loss / dataPoints.length;
  }

  get outputNodeId(): string {
    return this.graph.outputId;
  }

  regenerateData(): void {
    this.generateData();
    this.lossTrain = this.getLoss(this.trainData);
    this.lossTest = this.getLoss(this.testData);
  }

  rebuildNetwork(): void {
    this.graph = applyArchitecturePreset(this.config.architecturePreset, {
      networkShape: this.config.networkShape,
      numHiddenLayers: this.config.numHiddenLayers,
      activation: NETWORK_ACTIVATIONS[this.config.activation],
      outputActivation: this.outputActivation(),
      enabledFeatures: this.config.enabledFeatures,
    });
    // Apply the selected scheme (Link defaults are uniform); keep residual
    // identity skips (weight === 1 into sum nodes) intact.
    this.reinitializeWeights();
    this.applyRegularizationToLinks();
    this.syncOutputActivation();
  }

  setArchitecturePreset(preset: ArchitecturePresetId): void {
    this.config.architecturePreset = preset;
    this.reset();
  }

  markCustomArchitecture(): void {
    if (this.config.architecturePreset !== "custom") {
      this.config.architecturePreset = "custom";
    }
  }

  addPaletteNode(kind: GraphNodeKind, position: GraphPosition): GraphNode | null {
    if (kind === "input") return null;
    this.markCustomArchitecture();

    const activation =
      kind === "output" ? this.outputActivation()
      : kind === "sum" ? Activations.RELU
      : NETWORK_ACTIVATIONS[this.config.activation];

    const node = this.graph.addNode(kind, activation, { position, label: kind === "dense" ? "Dense" : kind === "sum" ? "Add" : "Out" });

    if (kind === "output" && this.graph.outputId !== node.id) {
      const prevOutput = [...this.graph.nodes.values()].find((n) => n.kind === "output" && n.id !== node.id);
      if (prevOutput) {
        for (const link of [...prevOutput.inputLinks]) {
          this.graph.connect(link.source.id, node.id, link.weight);
        }
        this.graph.removeNode(prevOutput.id);
      }
    }

    this.syncInputNodes();
    this.reinitAfterTopologyChange();
    return node;
  }

  connectNodes(sourceId: string, targetId: string): boolean {
    const link = this.graph.connect(sourceId, targetId);
    if (!link) return false;
    const dest = link.dest as GraphNode;
    const fanIn = Math.max(dest.inputLinks.length, 1);
    const fanOut = Math.max(dest.outputs.length, 1);
    link.weight = sampleWeight(this.config.weightInit, fanIn, fanOut);
    this.markCustomArchitecture();
    this.reinitAfterTopologyChange();
    return true;
  }

  disconnectNodes(sourceId: string, targetId: string): boolean {
    const source = this.graph.getNode(sourceId);
    if (!source) return false;
    const link = source.outputs.find((l) => l.dest.id === targetId);
    if (!link) return false;
    this.graph.removeLink(link.id);
    this.markCustomArchitecture();
    this.reinitAfterTopologyChange();
    return true;
  }

  removeGraphNode(nodeId: string): boolean {
    const node = this.graph.getNode(nodeId);
    if (!node || node.kind === "input") return false;
    // The output node is the loss/boundary anchor; removing it would leave the
    // graph invalid and crash training. Replace it by dropping a new Output block.
    if (node.kind === "output") return false;
    this.graph.removeNode(nodeId);
    this.markCustomArchitecture();
    this.syncInputNodes();
    this.reinitAfterTopologyChange();
    return true;
  }

  setNodePosition(nodeId: string, position: GraphPosition): void {
    this.graph.setPosition(nodeId, position);
  }

  /** Snap all nodes back to the canonical column grid (preserves topology). */
  normalizeLayout(): void {
    const inputIds = constructInputIds(this.config.enabledFeatures);
    normalizeGraphLayout(this.graph, inputIds);
  }

  addNeuron(layerIdx: number): void {
    if (this.config.architecturePreset !== "mlp") {
      this.config.architecturePreset = "mlp";
    }
    if (layerIdx < 0 || layerIdx >= this.config.numHiddenLayers) return;
    if (this.config.networkShape[layerIdx] >= 16) return;
    this.config.networkShape[layerIdx]++;
    this.rebuildMlpPreservingLayout();
  }

  removeNeuron(layerIdx: number): void {
    if (this.config.architecturePreset !== "mlp") {
      this.config.architecturePreset = "mlp";
    }
    if (layerIdx < 0 || layerIdx >= this.config.numHiddenLayers) return;
    if (this.config.networkShape[layerIdx] <= 1) return;
    this.config.networkShape[layerIdx]--;
    this.rebuildMlpPreservingLayout();
  }

  addLayer(): void {
    if (this.config.architecturePreset !== "mlp") {
      this.config.architecturePreset = "mlp";
    }
    if (this.config.numHiddenLayers >= 8) return;
    this.config.networkShape[this.config.numHiddenLayers] = 4;
    this.config.numHiddenLayers++;
    this.rebuildMlpPreservingLayout();
  }

  removeLayer(): void {
    if (this.config.architecturePreset !== "mlp") {
      this.config.architecturePreset = "mlp";
    }
    if (this.config.numHiddenLayers <= 0) return;
    this.config.numHiddenLayers--;
    this.config.networkShape.splice(this.config.numHiddenLayers);
    this.rebuildMlpPreservingLayout();
  }

  toggleFeature(featureId: string): void {
    if (!(featureId in this.config.enabledFeatures)) return;
    if (this.config.dataMode === "1d" && (FEATURES_2D_ONLY as readonly string[]).includes(featureId)) {
      return;
    }
    const next = !this.config.enabledFeatures[featureId];
    const enabledCount = Object.values(this.config.enabledFeatures).filter(Boolean).length;
    if (!next && enabledCount <= 1) return;
    this.config.enabledFeatures = { ...this.config.enabledFeatures, [featureId]: next };
    this.boundaryNeedsInputRefresh = true;
    this.reset();
  }

  setDataset(dataset: NetworkAnyDatasetId): void {
    this.config.dataset = dataset;
    this.reset();
  }

  /** Switch between 2D heatmaps and 1D curves (rebuilds data + features). */
  setDataMode(dataMode: NetworkDataMode): void {
    if (this.config.dataMode === dataMode) return;
    this.config.dataMode = dataMode;
    const regression = this.config.problemType === "regression";
    if (dataMode === "1d") {
      this.config.dataset = regression
        ? DEFAULT_DATASET_1D_REGRESSION
        : DEFAULT_DATASET_1D_CLASSIFICATION;
      this.config.enabledFeatures = { ...DEFAULT_FEATURES_1D };
    } else {
      this.config.dataset = regression
        ? DEFAULT_DATASET_2D_REGRESSION
        : DEFAULT_DATASET_2D_CLASSIFICATION;
      this.config.enabledFeatures = { ...DEFAULT_FEATURES_2D };
    }
    this.syncOutputActivation();
    this.reset();
  }

  /** Classification vs regression — swaps to a matching default dataset. */
  setProblemType(problemType: NetworkProblemType): void {
    if (this.config.problemType === problemType) return;
    this.config.problemType = problemType;
    if (this.config.dataMode === "1d") {
      this.config.dataset =
        problemType === "regression"
          ? DEFAULT_DATASET_1D_REGRESSION
          : DEFAULT_DATASET_1D_CLASSIFICATION;
    } else {
      this.config.dataset =
        problemType === "regression"
          ? DEFAULT_DATASET_2D_REGRESSION
          : DEFAULT_DATASET_2D_CLASSIFICATION;
    }
    this.syncOutputActivation();
    this.reset();
  }

  /**
   * Swap the hidden-unit activation in place, preserving learned weights and
   * any custom topology. Output and sum nodes keep their own activations.
   */
  setActivation(activation: NetworkActivationId): void {
    this.config.activation = activation;
    const fn = NETWORK_ACTIVATIONS[activation];
    for (const node of this.graph.nodes.values()) {
      if (node.kind === "dense") node.activation = fn;
    }
    this.refreshMetrics();
    this.boundaryNeedsInputRefresh = true;
    this.refreshBoundaryPreview();
  }

  /** Change weight init scheme and re-sample weights (epoch resets). */
  setWeightInit(weightInit: WeightInitId): void {
    this.config.weightInit = weightInit;
    this.resetWeights();
  }

  /** Switch optimizer; clears moment buffers so Adam/RMSProp start fresh. */
  setOptimizer(optimizer: PlaygroundOptimizerId): void {
    if (this.config.optimizer === optimizer) return;
    this.config.optimizer = optimizer;
    this.clearOptimizerState();
  }

  private clearOptimizerState(): void {
    this.optStep = 0;
    for (const node of this.graph.nodes.values()) {
      node.m = 0;
      node.v = 0;
      for (const link of node.inputLinks) {
        link.m = 0;
        link.v = 0;
      }
    }
  }

  /** Swap L1/L2/none on existing links without resetting weights. */
  setRegularization(regularization: NetworkRegularizationId): void {
    this.config.regularization = regularization;
    this.applyRegularizationToLinks();
  }

  /** Regenerate data with current settings, keeping network and progress. */
  updateDataParams(patch: Partial<Pick<NetworkPlaygroundConfig, "noise" | "percTrainData">>): void {
    Object.assign(this.config, patch);
    this.regenerateData();
  }

  private resolveRegularization(): RegFn | null {
    switch (this.config.regularization) {
      case "L1":
        return RegularizationFunction.L1;
      case "L2":
        return RegularizationFunction.L2;
      default:
        return null;
    }
  }

  /** Keep graph.regularization and every link in sync with config. */
  private applyRegularizationToLinks(): void {
    const fn = this.resolveRegularization();
    this.graph.regularization = fn;
    for (const link of this.graph.getAllLinks()) {
      link.regularization = fn;
      // L1 can mark weights dead; revive them when leaving L1.
      if (fn !== RegularizationFunction.L1) {
        link.isDead = false;
      }
    }
  }

  private outputActivation(): ActivationFunction {
    return this.config.problemType === "regression" ? Activations.LINEAR : Activations.TANH;
  }

  private syncOutputActivation(): void {
    const fn = this.outputActivation();
    for (const node of this.graph.nodes.values()) {
      if (node.kind === "output") node.activation = fn;
    }
  }

  private rebuildBoundaryStore(): void {
    if (this.config.dataMode === "1d") {
      const validation = this.graph.validate();
      this.curves = validation.valid
        ? initGraphCurveStore(this.graph)
        : {};
      this.refreshTargetCurve();
      this.boundary = {};
      return;
    }
    this.curves = {};
    this.targetCurve = null;
    const validation = this.graph.validate();
    if (!validation.valid) {
      this.boundary = computeBoundaries(this.network, this.config.enabledFeatures);
      return;
    }
    const preset = this.heatmap();
    this.boundary = initGraphBoundaryStore(this.graph, true, {
      output: preset.output,
      hidden: preset.hidden,
    });
  }

  private refreshTargetCurve(): void {
    if (this.config.dataMode !== "1d" || this.config.problemType !== "regression") {
      this.targetCurve = null;
      return;
    }
    if (!isDataset1DId(this.config.dataset)) {
      this.targetCurve = null;
      return;
    }
    this.targetCurve = targetCurve1D(this.config.dataset, CURVE_DENSITY);
  }

  private refreshBoundaryInternal(refreshInputFeatures: boolean): void {
    const validation = this.graph.validate();
    if (!validation.valid) {
      this.boundary = computeBoundaries(this.network, this.config.enabledFeatures);
      this.boundaryNeedsInputRefresh = false;
      return;
    }
    const preset = this.heatmap();
    updateGraphBoundaries(this.graph, this.config.enabledFeatures, this.boundary, {
      density: preset.output,
      hiddenDensity: preset.hidden,
      refreshInputFeatures: refreshInputFeatures || this.boundaryNeedsInputRefresh,
    });
    this.boundaryNeedsInputRefresh = false;
  }

  private refreshCurvesInternal(refreshInputFeatures: boolean): void {
    const validation = this.graph.validate();
    if (!validation.valid) {
      this.boundaryNeedsInputRefresh = false;
      return;
    }
    if (!Object.keys(this.curves).length) {
      this.curves = initGraphCurveStore(this.graph);
    }
    updateGraphCurves(this.graph, this.config.enabledFeatures, this.curves, {
      refreshInputFeatures: refreshInputFeatures || this.boundaryNeedsInputRefresh,
    });
    this.refreshTargetCurve();
    this.boundaryNeedsInputRefresh = false;
  }

  private syncInputNodes(): void {
    const inputIds = constructInputIds(this.config.enabledFeatures);
    this.graph.inputIds = inputIds;
    for (const id of inputIds) {
      if (!this.graph.getNode(id)) {
        this.graph.addInputNode(id);
      }
    }
    for (const [id, node] of this.graph.nodes) {
      if (node.kind === "input" && !inputIds.includes(id)) {
        this.graph.removeNode(id);
      }
    }
  }

  private rebuildMlpPreservingLayout(): void {
    const inputIds = constructInputIds(this.config.enabledFeatures);
    const shape = [
      inputIds.length,
      ...this.config.networkShape.slice(0, this.config.numHiddenLayers),
      1,
    ];
    this.graph = buildMlpGraph(
      shape,
      NETWORK_ACTIVATIONS[this.config.activation],
      this.outputActivation(),
      inputIds,
    );
    // Always snap to the canonical column grid (same as "Arrange layout") so
    // add/remove neuron/layer does not leave the new node dangling below the stack.
    normalizeGraphLayout(this.graph, inputIds);

    // The network is rebuilt with fresh weights, so training starts over.
    this.reinitializeWeights();
    this.applyRegularizationToLinks();
    this.epoch = 0;
    this.lossHistory = [];
    this.lossTrain = this.getLoss(this.trainData);
    this.lossTest = this.getLoss(this.testData);
    this.lossHistory.push({ epoch: 0, train: this.lossTrain, test: this.lossTest });
    this.boundaryNeedsInputRefresh = true;
    this.rebuildBoundaryStore();
    this.refreshBoundaryPreview();
  }

  private reinitAfterTopologyChange(): void {
    this.syncInputNodes();
    const validation = this.graph.validate();
    if (!validation.valid) return;

    this.lossTrain = this.getLoss(this.trainData);
    this.lossTest = this.getLoss(this.testData);
    this.boundaryNeedsInputRefresh = true;
    this.rebuildBoundaryStore();
    this.refreshBoundaryPreview();
  }

  private generateData(): void {
    if (this.config.dataMode === "1d") {
      const dataset = isDataset1DId(this.config.dataset)
        ? this.config.dataset
        : DEFAULT_DATASET_1D_CLASSIFICATION;
      this.config.dataset = dataset;
      const generator = DATASETS_1D[dataset];
      const data = generator(NUM_SAMPLES, this.config.noise / 100);
      shuffle(data);
      const splitIndex = Math.floor((data.length * this.config.percTrainData) / 100);
      this.trainData = data.slice(0, splitIndex);
      this.testData = data.slice(splitIndex);
      return;
    }
    const dataset = (this.config.dataset in DATASETS
      ? this.config.dataset
      : this.config.problemType === "regression"
        ? DEFAULT_DATASET_2D_REGRESSION
        : DEFAULT_DATASET_2D_CLASSIFICATION) as DatasetId;
    this.config.dataset = dataset;
    const generator: DataGenerator = DATASETS[dataset];
    const data = generator(NUM_SAMPLES, this.config.noise / 100);
    shuffle(data);
    const splitIndex = Math.floor((data.length * this.config.percTrainData) / 100);
    this.trainData = data.slice(0, splitIndex);
    this.testData = data.slice(splitIndex);
  }
}

export type { ArchitecturePresetId, GraphNodeKind, GraphPosition };
export type { HeatmapPresetId } from "./constants";
