import {
  IMAGE_SIZE,
  SIGNAL_LENGTH,
  imageToVolume,
  makeImageDataset,
  makeSignalDataset,
  signalToInput,
  type CnnDatasetId2D,
  type CnnDatasetId1D,
  type ImageExample,
  type SignalExample,
} from "./gallery";
import type { CnnActivationId } from "./activations";
import { Layer, type LayerShape } from "./layers/base";
import { Conv2DLayer } from "./layers/conv2d";
import { Pool2DLayer, type PoolKind2D } from "./layers/pool2d";
import { Conv1DLayer } from "./layers/conv1d";
import { Pool1DLayer } from "./layers/pool1d";
import { FlattenLayer } from "./layers/flatten";
import { DenseLayer } from "./layers/dense";
import { OutputLayer } from "./layers/output";
import { Losses } from "./loss";
import type { Signal, Volume } from "./tensor";

export type CnnMode = "2d" | "1d";

/** Declarative spec for one layer — the editable description of the network. */
export interface LayerSpec {
  kind: "conv2d" | "pool2d" | "conv1d" | "pool1d" | "flatten" | "dense";
  filters?: number;
  kernelSize?: number;
  poolKind?: PoolKind2D;
  units?: number;
  activation?: CnnActivationId;
}

export interface CnnConfig {
  mode: CnnMode;
  /** Dataset id (must match `mode`). */
  dataset: CnnDatasetId2D | CnnDatasetId1D;
  /** Ordered layer specs (input + output are implicit). */
  layers: LayerSpec[];
  learningRate: number;
  activation: CnnActivationId;
  batchSize: number;
  /** Noise level ∈ [0,1] passed to the dataset generator. */
  noise: number;
  percTrainData: number;
}

export const DEFAULT_CNN_CONFIG_2D: CnnConfig = {
  mode: "2d",
  dataset: "digits",
  layers: [
    { kind: "conv2d", filters: 4, kernelSize: 3, activation: "relu" },
    { kind: "pool2d", poolKind: "max" },
    { kind: "conv2d", filters: 8, kernelSize: 3, activation: "relu" },
    { kind: "pool2d", poolKind: "max" },
    { kind: "flatten" },
    { kind: "dense", units: 1, activation: "linear" },
  ],
  learningRate: 0.1,
  activation: "relu",
  batchSize: 16,
  noise: 0.1,
  percTrainData: 50,
};

export const DEFAULT_CNN_CONFIG_1D: CnnConfig = {
  mode: "1d",
  dataset: "heartbeat",
  layers: [
    { kind: "conv1d", filters: 4, kernelSize: 5, activation: "relu" },
    { kind: "pool1d", poolKind: "max" },
    { kind: "conv1d", filters: 8, kernelSize: 5, activation: "relu" },
    { kind: "pool1d", poolKind: "max" },
    { kind: "flatten" },
    { kind: "dense", units: 1, activation: "linear" },
  ],
  learningRate: 0.1,
  activation: "relu",
  batchSize: 16,
  noise: 0.1,
  percTrainData: 50,
};

export const DEFAULT_CNN_CONFIG: Record<CnnMode, CnnConfig> = {
  "2d": DEFAULT_CNN_CONFIG_2D,
  "1d": DEFAULT_CNN_CONFIG_1D,
};

let nextLayerId = 1;

export interface TrainingStats {
  epoch: number;
  lossTrain: number;
  lossTest: number;
  accTrain: number;
  accTest: number;
}

export interface FeatureMapSnapshot {
  layerId: string;
  /** For 2-D layers: one map per channel. For 1-D layers: empty. */
  maps2d?: number[][][];
  /** For 1-D layers: one row per channel. */
  signals?: number[][];
  /** Dense weight matrix `W[out][in]` for matrix visualization. */
  matrix?: number[][];
}

/**
 * Convolutional-network engine. Holds a layer pipeline built from a
 * {@link CnnConfig} and runs SGD with batched updates. The same instance serves
 * both modes; switching mode rebuilds the pipeline from the matching default.
 */
export class CnnEngine {
  config: CnnConfig;
  layers: Layer[] = [];
  trainData: ImageExample[] | SignalExample[] = [];
  testData: ImageExample[] | SignalExample[] = [];
  epoch = 0;
  lossTrain = 0;
  lossTest = 0;
  accTrain = 0;
  accTest = 0;
  /** Index of the example currently shown in the input node. */
  inspectedExampleIndex = 0;
  private readonly initialConfig: CnnConfig;

  constructor(config: CnnConfig) {
    this.config = structuredClone(config);
    this.initialConfig = structuredClone(config);
    this.bootstrap();
  }

  // ─── Lifecycle ──────────────────────────────────────────────────────────────

  private bootstrap(): void {
    this.epoch = 0;
    this.buildPipeline();
    this.generateData();
    this.refreshMetrics();
    this.forwardInspected();
  }

  /** Build the concrete layer pipeline (input + spec layers + output) from config. */
  private buildPipeline(): void {
    nextLayerId = 1;
    const layers: Layer[] = [];
    const mode = this.config.mode;
    if (mode === "2d") {
      layers.push(
        new (class extends Layer {
          constructor() {
            super("in", "input", "2d");
          }
          label() {
            return `Input ${IMAGE_SIZE}×${IMAGE_SIZE}`;
          }
          outputShape(): LayerShape {
            return { kind: "2d", channels: 1, rows: IMAGE_SIZE, cols: IMAGE_SIZE };
          }
          forward(input: Volume) {
            this.output = input;
            return input;
          }
          backward(gradOut: Volume) {
            this.inputGrad = gradOut;
            return gradOut;
          }
          updateParams() {}
          zeroGrads() {}
          paramCount() {
            return 0;
          }
          reinitialize() {}
          weightMagnitude() {
            return null;
          }
        })(),
      );
    } else {
      layers.push(
        new (class extends Layer {
          constructor() {
            super("in", "input", "1d");
          }
          label() {
            return `Input ${SIGNAL_LENGTH}`;
          }
          outputShape(): LayerShape {
            return { kind: "1d", channels: 1, length: SIGNAL_LENGTH };
          }
          forward(input: Signal) {
            this.output = input;
            return input;
          }
          backward(gradOut: Signal) {
            this.inputGrad = gradOut;
            return gradOut;
          }
          updateParams() {}
          zeroGrads() {}
          paramCount() {
            return 0;
          }
          reinitialize() {}
          weightMagnitude() {
            return null;
          }
        })(),
      );
    }

    for (const spec of this.config.layers) {
      const id = `L${nextLayerId++}`;
      switch (spec.kind) {
        case "conv2d": {
          // "Same" padding so convolutions preserve spatial size (odd kernels,
          // stride 1) — only pooling downsamples. This keeps the interactive
          // editor from collapsing dimensions below a kernel size.
          const k2 = spec.kernelSize ?? 3;
          layers.push(
            new Conv2DLayer(id, {
              filters: spec.filters ?? 4,
              kernelSize: k2,
              stride: 1,
              padding: Math.floor((k2 - 1) / 2),
              activation: spec.activation ?? this.config.activation,
            }),
          );
          break;
        }
        case "pool2d":
          layers.push(new Pool2DLayer(id, spec.poolKind ?? "max", 2));
          break;
        case "conv1d": {
          const k1 = spec.kernelSize ?? 5;
          layers.push(
            new Conv1DLayer(id, {
              filters: spec.filters ?? 4,
              kernelSize: k1,
              stride: 1,
              padding: Math.floor((k1 - 1) / 2),
              activation: spec.activation ?? this.config.activation,
            }),
          );
          break;
        }
        case "pool1d":
          layers.push(new Pool1DLayer(id, spec.poolKind ?? "max", 2));
          break;
        case "flatten":
          layers.push(new FlattenLayer(id));
          break;
        case "dense":
          layers.push(
            new DenseLayer(id, spec.units ?? 1, spec.activation ?? "linear"),
          );
          break;
      }
    }

    layers.push(new OutputLayer(`L${nextLayerId++}`, Losses.BINARY_CROSS_ENTROPY));
    this.layers = layers;
  }

  // ─── Data ───────────────────────────────────────────────────────────────────

  generateData(): void {
    const noise = this.config.noise;
    if (this.config.mode === "2d") {
      const data = makeImageDataset(this.config.dataset as CnnDatasetId2D, undefined, noise);
      this.splitData(data);
    } else {
      const data = makeSignalDataset(this.config.dataset as CnnDatasetId1D, undefined, noise);
      this.splitData(data);
    }
  }

  private splitData<T extends ImageExample | SignalExample>(data: T[]): void {
    const split = Math.floor((data.length * this.config.percTrainData) / 100);
    this.trainData = data.slice(0, split) as ImageExample[] | SignalExample[];
    this.testData = data.slice(split) as ImageExample[] | SignalExample[];
  }

  regenerateData(): void {
    this.generateData();
    this.refreshMetrics();
    this.forwardInspected();
  }

  updateDataParams(patch: Partial<Pick<CnnConfig, "noise" | "percTrainData">>): void {
    Object.assign(this.config, patch);
    this.regenerateData();
  }

  // ─── Forward / backward ─────────────────────────────────────────────────────

  /** Forward an example through the whole pipeline; returns the output layer. */
  private forwardExample(example: ImageExample | SignalExample): OutputLayer {
    const input =
      this.config.mode === "2d"
        ? imageToVolume((example as ImageExample).pixels)
        : signalToInput((example as SignalExample).values);
    let activation: Volume | Signal = input;
    for (const layer of this.layers) {
      activation = layer.forward(activation);
    }
    return this.outputLayer;
  }

  private get outputLayer(): OutputLayer {
    return this.layers[this.layers.length - 1] as OutputLayer;
  }

  /** Backprop one example (output → input); assumes a forward pass just ran. */
  private backwardExample(): void {
    const out = this.outputLayer;
    let grad: Volume | Signal = [new Array(1).fill(0)];
    grad = out.backward(grad as Signal);
    for (let i = this.layers.length - 2; i >= 0; i--) {
      grad = this.layers[i].backward(grad);
    }
  }

  /** One epoch: mini-batch SGD over the training set (averaged grads per batch). */
  trainEpoch(): void {
    const { batchSize, learningRate } = this.config;
    const data = this.trainData;
    let inBatch = 0;
    const flushBatch = () => {
      if (inBatch > 0) {
        this.applyUpdate(learningRate, inBatch);
        inBatch = 0;
      }
    };
    for (let i = 0; i < data.length; i++) {
      if (inBatch === 0) {
        for (const layer of this.layers) layer.zeroGrads();
      }
      const example = data[i];
      this.forwardExample(example);
      this.outputLayer.setTarget(example.label);
      this.backwardExample();
      inBatch++;
      if (inBatch >= batchSize) flushBatch();
    }
    flushBatch(); // trailing partial batch
    this.epoch++;
  }

  /**
   * Apply accumulated parameter gradients. Each layer already stores the
   * *per-example* grad from its most recent backward pass; we average by the
   * batch size by scaling the learning rate here.
   */
  private applyUpdate(learningRate: number, batchSize: number): void {
    const effectiveLr = learningRate / batchSize;
    for (const layer of this.layers) {
      layer.updateParams(effectiveLr);
    }
  }

  /** Predict the probability for an example. */
  predict(example: ImageExample | SignalExample): number {
    this.forwardExample(example);
    return this.outputLayer.probability;
  }

  getLoss(data: ImageExample[] | SignalExample[]): number {
    if (data.length === 0) return 0;
    let total = 0;
    for (const ex of data) {
      this.forwardExample(ex);
      total += this.outputLayer.loss(ex.label);
    }
    return total / data.length;
  }

  getAccuracy(data: ImageExample[] | SignalExample[]): number {
    if (data.length === 0) return 0;
    let correct = 0;
    for (const ex of data) {
      this.forwardExample(ex);
      const predicted = this.outputLayer.probability >= 0.5 ? 1 : 0;
      if (predicted === ex.label) correct++;
    }
    return correct / data.length;
  }

  refreshMetrics(): void {
    this.lossTrain = this.getLoss(this.trainData);
    this.lossTest = this.getLoss(this.testData);
    this.accTrain = this.getAccuracy(this.trainData);
    this.accTest = this.getAccuracy(this.testData);
  }

  /** Recompute feature maps for the inspected example (cheap, for display). */
  forwardInspected(): void {
    const data = this.trainData.length ? this.trainData : this.testData;
    if (!data.length) return;
    const idx = Math.min(this.inspectedExampleIndex, data.length - 1);
    this.forwardExample(data[idx]);
  }

  setInspectedExample(index: number): void {
    const data = this.trainData.length ? this.trainData : this.testData;
    this.inspectedExampleIndex = Math.max(0, Math.min(index, data.length - 1));
    this.forwardInspected();
  }

  // ─── Visualization snapshots ────────────────────────────────────────────────

  /** Snapshot of every layer's current output (the inspected example). */
  featureMaps(): FeatureMapSnapshot[] {
    this.forwardInspected();
    return this.layers.map((layer) => this.snapshotLayer(layer));
  }

  private snapshotLayer(layer: Layer): FeatureMapSnapshot {
    if (layer.dataSpace === "2d") {
      const vol = (layer.output as Volume) ?? [];
      const maps2d = vol.map((ch) => ch.map((row) => row.slice()));
      return { layerId: layer.id, maps2d };
    }
    const sig = (layer.output as Signal) ?? [];
    const snap: FeatureMapSnapshot = {
      layerId: layer.id,
      signals: sig.map((row) => row.slice()),
    };
    if (layer instanceof DenseLayer && layer.weights.length > 0) {
      snap.matrix = layer.weights.map((row) => row.slice());
    }
    return snap;
  }

  /** Per-layer display kernels for conv layers (sum over input channels). */
  kernelSnapshots(): Record<string, number[][] | number[][][]> {
    const out: Record<string, number[][] | number[][][]> = {};
    for (const layer of this.layers) {
      if (layer instanceof Conv2DLayer) out[layer.id] = layer.featureKernels();
      if (layer instanceof Conv1DLayer) out[layer.id] = layer.featureKernels();
    }
    return out;
  }

  // ─── Mutators ───────────────────────────────────────────────────────────────

  reset(): void {
    this.config = structuredClone(this.initialConfig);
    this.bootstrap();
  }

  resetWeights(): void {
    this.epoch = 0;
    for (const layer of this.layers) layer.reinitialize(Math.random);
    this.refreshMetrics();
    this.forwardInspected();
  }

  setMode(mode: CnnMode): void {
    if (mode === this.config.mode) return;
    this.config = structuredClone(mode === "2d" ? DEFAULT_CNN_CONFIG_2D : DEFAULT_CNN_CONFIG_1D);
    this.bootstrap();
  }

  setDataset(dataset: CnnDatasetId2D | CnnDatasetId1D): void {
    this.config.dataset = dataset;
    this.epoch = 0;
    this.generateData();
    this.refreshMetrics();
    this.forwardInspected();
  }

  setActivation(activation: CnnActivationId): void {
    this.config.activation = activation;
    for (const spec of this.config.layers) {
      if (spec.kind === "conv2d" || spec.kind === "conv1d") spec.activation = activation;
    }
    for (const layer of this.layers) {
      if (layer instanceof Conv2DLayer) layer.activationId = activation;
      if (layer instanceof Conv1DLayer) layer.activationId = activation;
    }
    this.refreshMetrics();
    this.forwardInspected();
  }

  setLearningRate(lr: number): void {
    this.config.learningRate = lr;
  }

  setBatchSize(bs: number): void {
    this.config.batchSize = bs;
  }

  /** Rebuild pipeline + reinit from current layer specs. */
  private rebuildPipeline(): void {
    this.epoch = 0;
    this.buildPipeline();
    this.refreshMetrics();
    this.forwardInspected();
  }

  /**
   * Rebuild after an in-place spec edit (kernel size, pooling kind). Public so
   * the UI can trigger it directly after mutating a config layer spec.
   */
  rebuildAfterSpecEdit(_index: number): void {
    this.rebuildPipeline();
  }

  addLayer(spec: LayerSpec, at?: number): void {
    if (at === undefined) this.config.layers.push(spec);
    else this.config.layers.splice(at, 0, spec);
    this.rebuildPipeline();
  }

  removeLayer(index: number): void {
    if (index < 0 || index >= this.config.layers.length) return;
    this.config.layers.splice(index, 1);
    this.rebuildPipeline();
  }

  setLayerFilters(index: number, filters: number): void {
    const spec = this.config.layers[index];
    if (!spec || (spec.kind !== "conv2d" && spec.kind !== "conv1d")) return;
    spec.filters = filters;
    // Rebuild to reallocate kernel banks, then snapshot is fresh.
    this.rebuildPipeline();
  }

  setLayerUnits(index: number, units: number): void {
    const spec = this.config.layers[index];
    if (!spec || spec.kind !== "dense") return;
    spec.units = units;
    this.rebuildPipeline();
  }

  step(): void {
    this.trainEpoch();
    this.refreshMetrics();
    this.forwardInspected();
  }

  /** Compute the shape that flows through the pipeline (for display). */
  pipelineShapes(): LayerShape[] {
    const shapes: LayerShape[] = [];
    let current: LayerShape =
      this.config.mode === "2d"
        ? { kind: "2d", channels: 1, rows: IMAGE_SIZE, cols: IMAGE_SIZE }
        : { kind: "1d", channels: 1, length: SIGNAL_LENGTH };
    shapes.push(current);
    for (const layer of this.layers.slice(1)) {
      current = layer.outputShape(current);
      shapes.push(current);
    }
    return shapes;
  }

  stats(): TrainingStats {
    return {
      epoch: this.epoch,
      lossTrain: this.lossTrain,
      lossTest: this.lossTest,
      accTrain: this.accTrain,
      accTest: this.accTest,
    };
  }
}
