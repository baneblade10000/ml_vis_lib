import { describe, expect, it } from "vitest";
import { reduceMatrix } from "../../charts/mini-heatmap";
import { boundaryToGridPoints, computeBoundaries } from "./boundary";
import { valueToRgb, PALETTE_HIGH, PALETTE_LOW } from "./colors";
import { PlaygroundEngine } from "./engine";

describe("PlaygroundEngine", () => {
  it("trains and reduces loss on circle data", () => {
    const engine = new PlaygroundEngine({ dataset: "circle", networkShape: [4], numHiddenLayers: 1 });
    const initialLoss = engine.lossTrain;
    for (let i = 0; i < 20; i++) engine.step();
    expect(engine.epoch).toBe(20);
    expect(engine.lossTrain).toBeLessThanOrEqual(initialLoss);
    expect(engine.boundary[engine.outputNodeId]).toBeDefined();
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

describe("computeBoundaries", () => {
  it("returns per-node matrices", () => {
    const engine = new PlaygroundEngine({ networkShape: [3], numHiddenLayers: 1 });
    const boundary = computeBoundaries(engine.network, engine.config.enabledFeatures, 20);
    const outputId = engine.outputNodeId;
    expect(boundary[outputId].length).toBe(20);
    expect(boundary[outputId][0].length).toBe(20);
  });
});
