import { describe, expect, it } from "vitest";
import { reduceMatrix } from "../../charts/mini-heatmap";
import { boundaryToGridPoints, computeBoundaries } from "./boundary";
import { valueToRgb, valueToRgbZeroWhite, PALETTE_HIGH, PALETTE_LOW, PALETTE_MID, PALETTE_ZERO_WHITE } from "./colors";
import { CURVE_DENSITY, DENSITY, HEATMAP_PRESETS, NODE_BOUNDARY_DENSITY, PLAY_BOUNDARY_STRIDE, PLAY_DISPLAY_DENSITY } from "./constants";
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
    // (Link.dest is typed as the nn.Node base, but the graph stores GraphNodes which
    // carry a `kind` discriminator at runtime.)
    const intoHidden = withGrad.filter(
      (l) => (l.dest as unknown as { kind: string }).kind === "dense",
    );
    const intoOutput = withGrad.filter(
      (l) => (l.dest as unknown as { kind: string }).kind === "output",
    );
    expect(intoHidden.length).toBeGreaterThan(0);
    expect(intoOutput.length).toBeGreaterThan(0);
  });

  it("stores coarse boundary grids for hidden nodes and full res for output", () => {
    const engine = new PlaygroundEngine({ networkShape: [3], numHiddenLayers: 1 });
    const outputId = engine.graph.outputId;
    const hiddenId = [...engine.graph.nodes.keys()].find(
      (id) => id !== "x" && id !== "y" && id !== outputId,
    );
    expect(engine.boundary[outputId].length).toBe(DENSITY);
    expect(hiddenId && engine.boundary[hiddenId].length).toBe(NODE_BOUNDARY_DENSITY);
  });

  it("paused hidden density divides output density; play output divides the paused grid", () => {
    expect(DENSITY % NODE_BOUNDARY_DENSITY).toBe(0);
    expect(DENSITY / PLAY_BOUNDARY_STRIDE).toBe(PLAY_DISPLAY_DENSITY);
  });

  it("heatmap presets keep output/hidden/play grids divisible", () => {
    for (const preset of Object.values(HEATMAP_PRESETS)) {
      expect(preset.output % preset.hidden).toBe(0);
      expect(preset.output % preset.playOutput).toBe(0);
    }
  });

  it("setHeatmapPreset resizes output and hidden boundary stores", () => {
    const engine = new PlaygroundEngine({
      networkShape: [3],
      numHiddenLayers: 1,
      heatmapPreset: "low",
    });
    const outputId = engine.graph.outputId;
    const hiddenId = [...engine.graph.nodes.keys()].find(
      (id) => id !== "x" && id !== "y" && id !== outputId,
    );
    expect(engine.boundary[outputId].length).toBe(HEATMAP_PRESETS.low.output);
    expect(hiddenId && engine.boundary[hiddenId].length).toBe(HEATMAP_PRESETS.low.hidden);
    engine.setHeatmapPreset("medium");
    expect(engine.boundary[outputId].length).toBe(HEATMAP_PRESETS.medium.output);
    expect(hiddenId && engine.boundary[hiddenId].length).toBe(HEATMAP_PRESETS.medium.hidden);
    engine.refreshOutputBoundaryFast();
    engine.refreshHiddenBoundariesFast();
    expect(engine.boundary[outputId].length).toBe(HEATMAP_PRESETS.medium.output);
    expect(hiddenId && engine.boundary[hiddenId].length).toBe(HEATMAP_PRESETS.medium.hidden);
  });

  it("play boundary refresh keeps paused store sizes", () => {
    const engine = new PlaygroundEngine({ networkShape: [3], numHiddenLayers: 1 });
    const outputId = engine.graph.outputId;
    const hiddenId = [...engine.graph.nodes.keys()].find(
      (id) => id !== "x" && id !== "y" && id !== outputId,
    );
    engine.refreshOutputBoundaryFast();
    engine.refreshHiddenBoundariesFast();
    expect(engine.boundary[outputId].length).toBe(DENSITY);
    expect(hiddenId && engine.boundary[hiddenId].length).toBe(NODE_BOUNDARY_DENSITY);
  });

  it("addNeuron fills the new hidden heatmap without a full-resolution sweep", () => {
    const engine = new PlaygroundEngine({ networkShape: [3], numHiddenLayers: 1 });
    engine.addNeuron(0);
    const hidden = [...engine.graph.nodes.values()].filter((n) => n.kind === "dense");
    expect(hidden).toHaveLength(4);
    for (const node of hidden) {
      const grid = engine.boundary[node.id];
      expect(grid.length).toBe(NODE_BOUNDARY_DENSITY);
      expect(grid[0]!.some((v) => Number.isFinite(v))).toBe(true);
    }
    expect(engine.boundary[engine.outputNodeId].length).toBe(DENSITY);
  });

  it("rebuilds boundary keys after applyTopologySnapshot (worker init path)", () => {
    const main = new PlaygroundEngine({ dataset: "circle", networkShape: [4], numHiddenLayers: 1 });
    main.addNeuron(0);
    main.addNeuron(0);
    main.addLayer();
    const snap = main.graph.toSnapshot();
    const worker = new PlaygroundEngine({ dataset: "xor", networkShape: [2], numHiddenLayers: 1 });
    // Simulate the old bug: bootstrap store for a smaller net, then swap topology.
    worker.applyTopologySnapshot(snap, {
      trainData: main.trainData,
      testData: main.testData,
    });
    for (const id of main.graph.nodes.keys()) {
      expect(worker.boundary[id], `missing boundary for ${id}`).toBeDefined();
    }
    expect(worker.boundary[worker.outputNodeId]?.length).toBe(DENSITY);
    expect(worker.trainData).toEqual(main.trainData);
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
    expect(valueToRgb(0)).toEqual(PALETTE_MID);
    expect(valueToRgb(1)).toEqual(PALETTE_HIGH);
  });

  it("valueToRgb keeps ±0.5 closer to midpoint than a linear ramp", () => {
    const dist = (
      a: { r: number; g: number; b: number },
      b: { r: number; g: number; b: number },
    ) => Math.hypot(a.r - b.r, a.g - b.g, a.b - b.b);
    const halfToMid = dist(valueToRgb(0.5), PALETTE_MID);
    const endToMid = dist(PALETTE_HIGH, PALETTE_MID);
    expect(halfToMid).toBeLessThan(endToMid * 0.4);
    expect(dist(valueToRgb(-0.5), PALETTE_MID)).toBeLessThan(dist(PALETTE_LOW, PALETTE_MID) * 0.4);
  });

  it("valueToRgbZeroWhite maps 0 to white", () => {
    expect(valueToRgbZeroWhite(-1)).toEqual(PALETTE_LOW);
    expect(valueToRgbZeroWhite(0)).toEqual(PALETTE_ZERO_WHITE);
    expect(valueToRgbZeroWhite(1)).toEqual(PALETTE_HIGH);
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
    expect(engine.curves[engine.outputNodeId]?.length).toBe(CURVE_DENSITY);
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
    expect(engine.targetCurve?.length).toBe(CURVE_DENSITY);
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
    // 2x boundary resolution (DENSITY 200→400) makes each step's full boundary
    // refresh ~4x heavier, so this real-training test needs a larger budget.
  }, 20000);
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
