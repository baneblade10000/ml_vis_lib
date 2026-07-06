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
  NUM_SAMPLES,
  shuffle,
  type DataGenerator,
  type DatasetId,
  type Example2D,
} from "./dataset";
import { computeBoundaries } from "./boundary";
import { constructInput, constructInputIds } from "./inputs";
import {
  Activations,
  Errors,
  type ActivationFunction,
  type Node,
} from "./nn";
import {
  applyArchitecturePreset,
  backPropGraph,
  buildMlpGraph,
  ComputationalGraph,
  forwardPropGraph,
  GraphNode,
  initGraphBoundaryStore,
  updateGraphBoundaries,
  updateGraphHiddenBoundaries,
  updateGraphOutputBoundary,
  updateWeightsGraph,
  type ArchitecturePresetId,
  type GraphNodeKind,
  type GraphPosition,
} from "./graph";
import { layoutMlpFromLayers, MLP_ROW_SPACING, normalizeGraphLayout } from "./graph/mlp-layout";
import { PLAY_BOUNDARY_STRIDE } from "./constants";

export type TfActivationId = "relu" | "tanh" | "sigmoid" | "linear";

export const TF_ACTIVATIONS: Record<TfActivationId, ActivationFunction> = {
  relu: Activations.RELU,
  tanh: Activations.TANH,
  sigmoid: Activations.SIGMOID,
  linear: Activations.LINEAR,
};

export interface TfPlaygroundConfig {
  learningRate: number;
  activation: TfActivationId;
  batchSize: number;
  noise: number;
  percTrainData: number;
  dataset: DatasetId;
  networkShape: number[];
  numHiddenLayers: number;
  enabledFeatures: Record<string, boolean>;
  regularizationRate?: number;
  discretize?: boolean;
  architecturePreset: ArchitecturePresetId;
}

export const DEFAULT_TF_CONFIG: TfPlaygroundConfig = {
  learningRate: 0.03,
  activation: "tanh",
  batchSize: 10,
  noise: 0,
  percTrainData: 50,
  dataset: "circle",
  networkShape: [2],
  numHiddenLayers: 1,
  enabledFeatures: { x: true, y: true, xSquared: false, ySquared: false, xTimesY: false, sinX: false, sinY: false },
  regularizationRate: 0,
  discretize: false,
  architecturePreset: "mlp",
};

export interface LossHistoryPoint {
  epoch: number;
  train: number;
  test: number;
}

export class PlaygroundEngine {
  config: TfPlaygroundConfig;
  graph: ComputationalGraph = new ComputationalGraph();
  boundary: Record<string, number[][]> = {};
  trainData: Example2D[] = [];
  testData: Example2D[] = [];
  lossTrain = 0;
  lossTest = 0;
  epoch = 0;
  lossHistory: LossHistoryPoint[] = [];
  private boundaryNeedsInputRefresh = true;
  /** Config snapshot from first page load — used by resetToInitial(). */
  private readonly initialConfig: TfPlaygroundConfig;

  constructor(config: Partial<TfPlaygroundConfig> = {}) {
    this.config = this.cloneConfig(this.mergeConfig(config));
    this.initialConfig = this.cloneConfig(this.config);
    this.bootstrap();
  }

  private mergeConfig(config: Partial<TfPlaygroundConfig>): TfPlaygroundConfig {
    return {
      ...DEFAULT_TF_CONFIG,
      ...config,
      networkShape: [...(config.networkShape ?? DEFAULT_TF_CONFIG.networkShape)],
      enabledFeatures: { ...(config.enabledFeatures ?? DEFAULT_TF_CONFIG.enabledFeatures) },
    };
  }

  private cloneConfig(config: TfPlaygroundConfig): TfPlaygroundConfig {
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

  /** Restore config, data, and network exactly as on first page load. */
  resetToInitial(): void {
    this.config = this.cloneConfig(this.initialConfig);
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

  /** Fresh random weights / biases in place, keeping the topology. */
  private reinitializeWeights(): void {
    for (const node of this.graph.nodes.values()) {
      if (node.kind === "input") continue;
      node.bias = node.kind === "sum" ? 0 : 0.1;
      for (const link of node.inputLinks) {
        link.weight = Math.random() - 0.5;
        link.isDead = false;
      }
    }
  }

  step(): void {
    this.trainEpoch();
    this.refreshMetrics();
    this.refreshBoundary();
    this.lossHistory.push({ epoch: this.epoch, train: this.lossTrain, test: this.lossTest });
  }

  /** One training pass without boundary / test-loss refresh (for fast Play loop). */
  trainEpoch(updateTrainLoss = false): void {
    this.epoch++;
    const { batchSize, learningRate, regularizationRate = 0 } = this.config;
    this.trainData.forEach((point, i) => {
      const input = constructInput(point.x, point.y, this.config.enabledFeatures);
      forwardPropGraph(this.graph, input, i === 0);
      backPropGraph(this.graph, point.label, Errors.SQUARE);
      if ((i + 1) % batchSize === 0) {
        updateWeightsGraph(this.graph, learningRate, regularizationRate);
      }
    });
    if (updateTrainLoss) {
      this.lossTrain = this.getLoss(this.trainData);
    }
  }

  refreshMetrics(): void {
    this.lossTrain = this.getLoss(this.trainData);
    this.lossTest = this.getLoss(this.testData);
  }

  /** Full-quality boundary for all nodes. */
  refreshBoundary(): void {
    this.refreshBoundaryInternal(this.boundaryNeedsInputRefresh);
  }

  /** Fast output-only boundary during Play (stride subsampling). */
  refreshOutputBoundaryFast(stride = PLAY_BOUNDARY_STRIDE): void {
    const validation = this.graph.validate();
    if (!validation.valid) return;
    updateGraphOutputBoundary(
      this.graph,
      this.config.enabledFeatures,
      this.boundary,
      this.graph.outputId,
      stride,
    );
  }

  /** Fast coarse refresh of hidden-node heatmaps during Play (10×10 grid). */
  refreshHiddenBoundariesFast(): void {
    const validation = this.graph.validate();
    if (!validation.valid) return;
    updateGraphHiddenBoundaries(
      this.graph,
      this.config.enabledFeatures,
      this.boundary,
      this.graph.outputId,
    );
  }

  getLoss(dataPoints: Example2D[] = this.trainData): number {
    if (!dataPoints.length) return 0;
    let loss = 0;
    for (const dataPoint of dataPoints) {
      const input = constructInput(dataPoint.x, dataPoint.y, this.config.enabledFeatures);
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
      activation: TF_ACTIVATIONS[this.config.activation],
      outputActivation: Activations.TANH,
      enabledFeatures: this.config.enabledFeatures,
    });
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
      kind === "output" ? Activations.TANH
      : kind === "sum" ? Activations.RELU
      : TF_ACTIVATIONS[this.config.activation];

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
    if (this.config.networkShape[layerIdx] >= 8) return;
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
    if (this.config.numHiddenLayers >= 6) return;
    this.config.networkShape[this.config.numHiddenLayers] = 2;
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
    const next = !this.config.enabledFeatures[featureId];
    const enabledCount = Object.values(this.config.enabledFeatures).filter(Boolean).length;
    if (!next && enabledCount <= 1) return;
    this.config.enabledFeatures = { ...this.config.enabledFeatures, [featureId]: next };
    this.boundaryNeedsInputRefresh = true;
    this.reset();
  }

  setDataset(dataset: DatasetId): void {
    this.config.dataset = dataset;
    this.reset();
  }

  /**
   * Swap the hidden-unit activation in place, preserving learned weights and
   * any custom topology. Output and sum nodes keep their own activations.
   */
  setActivation(activation: TfActivationId): void {
    this.config.activation = activation;
    const fn = TF_ACTIVATIONS[activation];
    for (const node of this.graph.nodes.values()) {
      if (node.kind === "dense") node.activation = fn;
    }
    this.refreshMetrics();
    this.boundaryNeedsInputRefresh = true;
    this.refreshBoundary();
  }

  /** Regenerate data with current settings, keeping network and progress. */
  updateDataParams(patch: Partial<Pick<TfPlaygroundConfig, "noise" | "percTrainData">>): void {
    Object.assign(this.config, patch);
    this.regenerateData();
  }

  private rebuildBoundaryStore(): void {
    const validation = this.graph.validate();
    if (!validation.valid) {
      this.boundary = computeBoundaries(this.network, this.config.enabledFeatures);
      return;
    }
    this.boundary = initGraphBoundaryStore(this.graph);
  }

  private refreshBoundaryInternal(refreshInputFeatures: boolean): void {
    const validation = this.graph.validate();
    if (!validation.valid) {
      this.boundary = computeBoundaries(this.network, this.config.enabledFeatures);
      this.boundaryNeedsInputRefresh = false;
      return;
    }
    updateGraphBoundaries(this.graph, this.config.enabledFeatures, this.boundary, {
      refreshInputFeatures: refreshInputFeatures || this.boundaryNeedsInputRefresh,
    });
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
    const oldLayers = this.graph.toLayeredNetwork();
    const posBySlot = new Map<string, GraphPosition>();
    for (let li = 0; li < oldLayers.length; li++) {
      for (let ri = 0; ri < oldLayers[li].length; ri++) {
        const pos = this.graph.positions.get(oldLayers[li][ri].id);
        if (pos) posBySlot.set(`${li}:${ri}`, { ...pos });
      }
    }

    const shape = [
      inputIds.length,
      ...this.config.networkShape.slice(0, this.config.numHiddenLayers),
      1,
    ];
    this.graph = buildMlpGraph(
      shape,
      TF_ACTIVATIONS[this.config.activation],
      Activations.TANH,
      inputIds,
    );

    layoutMlpFromLayers(this.graph, this.graph.toLayeredNetwork(), inputIds);

    const newLayers = this.graph.toLayeredNetwork();
    // Old positions are keyed by layer index, so they are only meaningful when
    // the layer count is unchanged (add/remove neuron). When a layer is added
    // or removed, the indices shift and restoring them would scramble columns —
    // keep the canonical layout instead.
    if (newLayers.length === oldLayers.length) {
      for (let li = 0; li < newLayers.length; li++) {
        const restored: GraphPosition[] = [];
        const fresh: GraphNode[] = [];
        for (let ri = 0; ri < newLayers[li].length; ri++) {
          const saved = posBySlot.get(`${li}:${ri}`);
          if (saved) {
            this.graph.setPosition(newLayers[li][ri].id, saved);
            restored.push(saved);
          } else {
            fresh.push(newLayers[li][ri]);
          }
        }
        // A brand-new neuron goes below the bottom of its column instead of
        // its canonical centered slot, which can land on a restored node.
        if (fresh.length > 0 && restored.length > 0) {
          const x = restored[0].x;
          let y = Math.max(...restored.map((p) => p.y));
          for (const node of fresh) {
            y += MLP_ROW_SPACING;
            this.graph.setPosition(node.id, { x, y });
          }
        }
      }
    }

    // The network is rebuilt with fresh weights, so training starts over.
    this.epoch = 0;
    this.lossHistory = [];
    this.lossTrain = this.getLoss(this.trainData);
    this.lossTest = this.getLoss(this.testData);
    this.boundaryNeedsInputRefresh = true;
    this.rebuildBoundaryStore();
    this.refreshBoundary();
  }

  private reinitAfterTopologyChange(): void {
    this.syncInputNodes();
    const validation = this.graph.validate();
    if (!validation.valid) return;

    this.lossTrain = this.getLoss(this.trainData);
    this.lossTest = this.getLoss(this.testData);
    this.boundaryNeedsInputRefresh = true;
    this.rebuildBoundaryStore();
    this.refreshBoundary();
  }

  private generateData(): void {
    const generator: DataGenerator = DATASETS[this.config.dataset];
    const noise = this.config.noise / 100;
    const data = generator(NUM_SAMPLES, noise);
    shuffle(data);
    const splitIndex = Math.floor((data.length * this.config.percTrainData) / 100);
    this.trainData = data.slice(0, splitIndex);
    this.testData = data.slice(splitIndex);
  }
}

export type { ArchitecturePresetId, GraphNodeKind, GraphPosition };
