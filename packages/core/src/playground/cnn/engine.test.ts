import { describe, expect, it } from "vitest";
import {
  CnnEngine,
  Conv2DLayer,
  Conv1DLayer,
  DenseLayer,
  OutputLayer,
  DEFAULT_CNN_CONFIG_2D,
  DEFAULT_CNN_CONFIG_1D,
  imageToVolume,
  signalToInput,
  type ImageExample,
  type SignalExample,
} from "./index";

/** Deterministic pseudo-random source so grad-checks are reproducible. */
function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Sum every element of a 2-D volume (channel-major). */
function sumVolume(output: number[][][]): number {
  let s = 0;
  for (const ch of output) {
    for (const row of ch) for (const v of row) s += v;
  }
  return s;
}

/** Sum every element of a 1-D signal (channel-major). */
function sumSignal(output: number[][]): number {
  let s = 0;
  for (const row of output) for (const v of row) s += v;
  return s;
}

function relErr(numeric: number, analytic: number): number {
  const denom = Math.max(1, Math.abs(numeric), Math.abs(analytic));
  return Math.abs(numeric - analytic) / denom;
}

/**
 * Numerical gradient check for a 2-D (volume) layer. Perturbs a few input cells
 * by ±eps and compares the finite-difference change in the summed output to the
 * analytic input gradient produced by the layer's backward pass.
 */
function gradCheckVolume(layer: Conv2DLayer, input: number[][][], eps = 1e-4): void {
  layer.forward(input);
  // Fixed upstream gradient of all-ones, sized to the output.
  const out = layer.output as unknown as number[][][];
  const gradOut: number[][][] = out.map((ch) => ch.map((row) => row.map(() => 1)));
  layer.backward(gradOut);
  const analytic = layer.inputGrad as unknown as number[][][];

  const cells: [number, number, number][] = [];
  for (let c = 0; c < input.length; c++)
    for (let r = 0; r < input[c].length; r++)
      for (let col = 0; col < input[c][r].length; col++) cells.push([c, r, col]);
  const sample = cells.filter((_, i) => i % Math.max(1, Math.floor(cells.length / 10)) === 0).slice(0, 10);

  for (const [c, r, col] of sample) {
    const original = input[c][r][col];
    input[c][r][col] = original + eps;
    layer.forward(input);
    const plus = sumVolume(layer.output as unknown as number[][][]);
    input[c][r][col] = original - eps;
    layer.forward(input);
    const minus = sumVolume(layer.output as unknown as number[][][]);
    input[c][r][col] = original;
    const numeric = (plus - minus) / (2 * eps);
    const a = analytic[c]?.[r]?.[col] ?? 0;
    expect(relErr(numeric, a)).toBeLessThan(1e-2);
  }
}

/** 1-D analogue of {@link gradCheckVolume} for conv1d/dense layers. */
function gradCheckSignal(layer: Conv1DLayer | DenseLayer, input: number[][], eps = 1e-4): void {
  layer.forward(input);
  const out = layer.output as unknown as number[][];
  const gradOut = out.map((row) => row.map(() => 1));
  layer.backward(gradOut);
  const analytic = layer.inputGrad as unknown as number[][];

  const cells: [number, number][] = [];
  for (let c = 0; c < input.length; c++) for (let p = 0; p < input[c].length; p++) cells.push([c, p]);
  const sample = cells.filter((_, i) => i % Math.max(1, Math.floor(cells.length / 10)) === 0).slice(0, 10);

  for (const [c, p] of sample) {
    const original = input[c][p];
    input[c][p] = original + eps;
    layer.forward(input);
    const plus = sumSignal(layer.output as unknown as number[][]);
    input[c][p] = original - eps;
    layer.forward(input);
    const minus = sumSignal(layer.output as unknown as number[][]);
    input[c][p] = original;
    const numeric = (plus - minus) / (2 * eps);
    const a = analytic[c]?.[p] ?? 0;
    expect(relErr(numeric, a)).toBeLessThan(1e-2);
  }
}

describe("CNN conv2d backprop", () => {
  it("matches finite-difference input gradient", () => {
    const rng = seeded(42);
    const layer = new Conv2DLayer("c", { filters: 2, kernelSize: 3, stride: 1, padding: 1, activation: "tanh" });
    layer.initParams(1, rng);
    const input = imageToVolume([[1, 2, 0, -1], [0, 3, 1, 2], [2, -1, 0, 1], [1, 1, -2, 0]]);
    gradCheckVolume(layer, input);
  });
});

describe("CNN conv1d backprop", () => {
  it("matches finite-difference input gradient", () => {
    const rng = seeded(7);
    const layer = new Conv1DLayer("c", { filters: 3, kernelSize: 3, stride: 1, padding: 1, activation: "relu" });
    layer.initParams(1, rng);
    const input = signalToInput([0.5, -0.5, 1, 0.2, -0.3, 0.8, -1, 0.1, 0.4, -0.6]);
    gradCheckSignal(layer, input);
  });
});

describe("CNN dense backprop", () => {
  it("matches finite-difference input gradient", () => {
    const rng = seeded(13);
    const layer = new DenseLayer("d", 2, "tanh");
    layer.initParams(4, rng);
    gradCheckSignal(layer, signalToInput([0.3, -0.7, 0.5, 0.9]));
  });
});

describe("CNN output layer", () => {
  it("produces a probability in (0,1) and a loss", () => {
    const rng = seeded(1);
    const out = new OutputLayer("o");
    out.initParams(3, rng);
    out.forward(signalToInput([0.2, -0.4, 0.6]));
    expect(out.probability).toBeGreaterThan(0);
    expect(out.probability).toBeLessThan(1);
    out.setTarget(1);
    expect(out.loss(1)).toBeGreaterThanOrEqual(0);
  });
});

describe("CnnEngine (2D)", () => {
  it("trains and reduces loss on the digits dataset", () => {
    const engine = new CnnEngine({ ...DEFAULT_CNN_CONFIG_2D, noise: 0.02 });
    const initialLoss = engine.lossTrain;
    for (let i = 0; i < 60; i++) engine.trainEpoch();
    engine.refreshMetrics();
    expect(engine.epoch).toBe(60);
    expect(engine.lossTrain).toBeLessThan(initialLoss);
    expect(engine.accTrain).toBeGreaterThan(0.7);
  });

  it("reports feature-map snapshots for every layer", () => {
    const engine = new CnnEngine({ ...DEFAULT_CNN_CONFIG_2D });
    const maps = engine.featureMaps();
    expect(maps.length).toBe(engine.layers.length);
    expect(maps[0].maps2d).toBeDefined();
    expect((maps[0].maps2d!.length)).toBe(1);
    const conv = maps.find((m) => m.kernels2d != null);
    expect(conv).toBeDefined();
    expect(conv!.biases).toBeDefined();
    expect(conv!.biases!.length).toBe(conv!.kernels2d!.length);
  });

  it("uses global average pooling instead of a second pool+flatten", () => {
    const engine = new CnnEngine({ ...DEFAULT_CNN_CONFIG_2D });
    const kinds = engine.layers.map((l) => l.kind);
    expect(kinds).toContain("gap2d");
    expect(kinds.filter((k) => k === "pool2d")).toHaveLength(1);
    expect(kinds).not.toContain("flatten");
    const shapes = engine.pipelineShapes();
    const gapIdx = engine.layers.findIndex((l) => l.kind === "gap2d");
    const gapShape = shapes[gapIdx]!;
    expect(gapShape.kind).toBe("1d");
    if (gapShape.kind === "1d") expect(gapShape.length).toBe(8);
  });

  it("reduces loss after training a few epochs on a 1D dataset", () => {
    const engine = new CnnEngine({ ...DEFAULT_CNN_CONFIG_1D, noise: 0.02 });
    const initialLoss = engine.lossTrain;
    for (let i = 0; i < 60; i++) engine.trainEpoch();
    engine.refreshMetrics();
    expect(engine.lossTrain).toBeLessThan(initialLoss);
    expect(engine.accTrain).toBeGreaterThan(0.7);
  });

  it("switches mode and rebuilds the pipeline", () => {
    const engine = new CnnEngine({ ...DEFAULT_CNN_CONFIG_2D });
    engine.setMode("1d");
    expect(engine.config.mode).toBe("1d");
    expect(engine.layers[0].dataSpace).toBe("1d");
    const maps = engine.featureMaps();
    expect(maps[0].signals).toBeDefined();
  });

  it("produces a non-degenerate test set under a 50% split", () => {
    const engine = new CnnEngine({ ...DEFAULT_CNN_CONFIG_2D });
    expect(engine.trainData.length).toBeGreaterThan(0);
    expect(engine.testData.length).toBeGreaterThan(0);
    const sample = (engine.testData[0] as ImageExample | SignalExample).label;
    expect(sample === 0 || sample === 1).toBe(true);
  });

  it("strong L1 drives dense weights toward zero", () => {
    const engine = new CnnEngine({
      ...DEFAULT_CNN_CONFIG_2D,
      regularization: "L1",
      regularizationRate: 10,
      learningRate: 0.3,
    });
    const dense = engine.layers.find((l) => l instanceof DenseLayer) as DenseLayer;
    expect(dense).toBeDefined();
    const before = dense.weights.flat().reduce((s, w) => s + Math.abs(w), 0);
    for (let i = 0; i < 40; i++) engine.trainEpoch();
    const after = dense.weights.flat().reduce((s, w) => s + Math.abs(w), 0);
    expect(after).toBeLessThan(before);
  });
});
