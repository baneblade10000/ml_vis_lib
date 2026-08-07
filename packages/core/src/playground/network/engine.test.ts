import { describe, expect, it } from "vitest";
import { reduceMatrix } from "../../charts/mini-heatmap";
import { boundaryToGridPoints, computeBoundaries } from "./boundary";
import { valueToRgb, PALETTE_HIGH, PALETTE_LOW } from "./colors";
import { PlaygroundEngine } from "./engine";
import { Activations } from "./nn";

describe("PlaygroundEngine", () => {
  it("trains and reduces loss on circle data", () => {
    const engine = new PlaygroundEngine({ dataset: "circle", networkShape: [4], numHiddenLayers: 1 });
    const initialLoss = engine.lossTrain;
    for (let i = 0; i < 20; i++) engine.step();
    expect(engine.epoch).toBe(20);
    expect(engine.lossTrain).toBeLessThanOrEqual(initialLoss);
    expect(engine.boundary[engine.outputNodeId]).toBeDefined();
  });

  it("keeps lastGradient after step() so multi-layer edge viz survives loss refresh", () => {
    const engine = new PlaygroundEngine({
      dataset: "circle",
      networkShape: [3, 3],
      numHiddenLayers: 2,
    });
    engine.step();
    const links = engine.graph.getAllLinks();
    expect(links.length).toBeGreaterThan(0);
    const withGrad = links.filter((l) => l.lastGradient !== 0);
    expect(withGrad.length).toBeGreaterThan(0);
    // Both early (into first hidden) and late (into output) edges should retain signal.
    const intoHidden = withGrad.filter((l) => l.dest.kind === "dense");
    const intoOutput = withGrad.filter((l) => l.dest.kind === "output");
    expect(intoHidden.length).toBeGreaterThan(0);
    expect(intoOutput.length).toBeGreaterThan(0);
  });

  it("stores coarse boundary grids for hidden nodes and full res for output", () => {
    const engine = new PlaygroundEngine({ networkShape: [3], numHiddenLayers: 1 });
    const outputId = engine.graph.outputId;
    const hiddenId = [...engine.graph.nodes.keys()].find(
      (id) => id !== "x" && id !== "y" && id !== outputId,
    );
    expect(engine.boundary[outputId].length).toBe(200);
    expect(hiddenId && engine.boundary[hiddenId].length).toBe(10);
  });

  it("maps boundary values to probabilities in [0, 1]", () => {
    const engine = new PlaygroundEngine();
    const matrix = engine.boundary[engine.outputNodeId];
    const grid = boundaryToGridPoints(matrix, [-6, 6], true);
    expect(grid.length).toBeGreaterThan(0);
    for (const point of grid) {
      expect(point.probability).toBeGreaterThanOrEqual(0);
      expect(point.probability).toBeLessThanOrEqual(1);
    }
  });
});

describe("mini-heatmap helpers", () => {
  it("reduceMatrix averages blocks", () => {
    const matrix = [
      [0, 1],
      [2, 3],
    ];
    expect(reduceMatrix(matrix, 2)[0][0]).toBe(1.5);
  });

  it("valueToRgb returns palette endpoints", () => {
    expect(valueToRgb(-1)).toEqual(PALETTE_LOW);
    expect(valueToRgb(1)).toEqual(PALETTE_HIGH);
  });
});

describe("weight initialization", () => {
  it("zeros init sets all weights and biases to zero", () => {
    const engine = new PlaygroundEngine({ weightInit: "zeros", networkShape: [3], numHiddenLayers: 1 });
    for (const node of engine.graph.nodes.values()) {
      if (node.kind === "input") continue;
      expect(node.bias).toBe(0);
      for (const link of node.inputLinks) {
        expect(link.weight).toBe(0);
      }
    }
  });

  it("setWeightInit re-samples weights and resets epoch", () => {
    const engine = new PlaygroundEngine({ weightInit: "uniform", networkShape: [2], numHiddenLayers: 1 });
    engine.step();
    expect(engine.epoch).toBe(1);
    engine.setWeightInit("zeros");
    expect(engine.config.weightInit).toBe("zeros");
    expect(engine.epoch).toBe(0);
    for (const node of engine.graph.nodes.values()) {
      if (node.kind === "input") continue;
      for (const link of node.inputLinks) {
        expect(link.weight).toBe(0);
      }
    }
  });
});

describe("1D data mode", () => {
  it("builds curve store and trains on gauss1d", () => {
    const engine = new PlaygroundEngine({
      dataMode: "1d",
      problemType: "classification",
      dataset: "gauss1d",
      networkShape: [4],
      numHiddenLayers: 1,
    });
    expect(engine.config.dataMode).toBe("1d");
    expect(Object.keys(engine.curves).length).toBeGreaterThan(0);
    expect(engine.curves[engine.outputNodeId]?.length).toBe(240);
    const initialLoss = engine.lossTrain;
    for (let i = 0; i < 25; i++) engine.step();
    expect(engine.lossTrain).toBeLessThanOrEqual(initialLoss + 1e-6);
  });

  it("setDataMode switches features and datasets", () => {
    const engine = new PlaygroundEngine();
    engine.setDataMode("1d");
    expect(engine.config.enabledFeatures.y).toBe(false);
    expect(engine.config.enabledFeatures.x).toBe(true);
    engine.setProblemType("regression");
    expect(engine.config.dataset).toBe("sine");
    expect(engine.targetCurve?.length).toBe(240);
    engine.setDataMode("2d");
    expect(engine.config.problemType).toBe("regression");
    expect(engine.config.dataset).toBe("sinSin");
    expect(engine.config.enabledFeatures.y).toBe(true);
  });

  it("resetToInitial keeps the active 1D layout", () => {
    const engine = new PlaygroundEngine();
    engine.setDataMode("1d");
    engine.setProblemType("regression");
    engine.setDataset("cubic");
    engine.resetToInitial();
    expect(engine.config.dataMode).toBe("1d");
    expect(engine.config.problemType).toBe("regression");
    expect(engine.config.dataset).toBe("cubic");
  });
});

describe("2D regression", () => {
  it("setProblemType enables sinSin with linear output", () => {
    const engine = new PlaygroundEngine();
    engine.setProblemType("regression");
    expect(engine.config.dataMode).toBe("2d");
    expect(engine.config.dataset).toBe("sinSin");
    expect(engine.graph.getOutputNode().activation).toBe(Activations.LINEAR);
    expect(engine.trainData.every((p) => Number.isFinite(p.label))).toBe(true);
  });

  it("trains on sin(x)·sin(y)", () => {
    const engine = new PlaygroundEngine({
      problemType: "regression",
      dataset: "sinSin",
      networkShape: [8, 8],
      numHiddenLayers: 2,
      learningRate: 0.1,
    });
    const initialLoss = engine.lossTrain;
    for (let i = 0; i < 40; i++) engine.step();
    expect(engine.lossTrain).toBeLessThan(initialLoss);
  });
});

describe("regularization", () => {
  it("setRegularization attaches L2 to every link", () => {
    const engine = new PlaygroundEngine({ networkShape: [2], numHiddenLayers: 1 });
    engine.setRegularization("L2");
    expect(engine.config.regularization).toBe("L2");
    expect(engine.graph.regularization).not.toBeNull();
    for (const link of engine.graph.getAllLinks()) {
      expect(link.regularization).toBe(engine.graph.regularization);
    }
  });

  it("strong L1 drives some weights to zero over training", () => {
    const engine = new PlaygroundEngine({
      networkShape: [4],
      numHiddenLayers: 1,
      regularization: "L1",
      regularizationRate: 10,
      learningRate: 0.3,
    });
    for (let i = 0; i < 40; i++) engine.step();
    const deadOrZero = engine.graph.getAllLinks().filter((l) => l.isDead || l.weight === 0);
    expect(deadOrZero.length).toBeGreaterThan(0);
  });
});

describe("computeBoundaries", () => {
  it("returns per-node matrices", () => {
    const engine = new PlaygroundEngine({ networkShape: [3], numHiddenLayers: 1 });
    const boundary = computeBoundaries(engine.network, engine.config.enabledFeatures, 20);
    const outputId = engine.outputNodeId;
    expect(boundary[outputId].length).toBe(20);
    expect(boundary[outputId][0].length).toBe(20);
  });
});
