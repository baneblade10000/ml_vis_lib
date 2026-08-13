import { describe, expect, it } from "vitest";
import { Activations } from "./nn";
import {
  applyArchitecturePreset,
  backPropGraph,
  buildMiniResNetGraph,
  buildResBlockGraph,
  forwardPropGraph,
  updateWeightsGraph,
} from "./graph";
import { constructInputIds } from "./inputs";
import { Errors } from "./nn";
import { PlaygroundEngine } from "./engine";
import { MLP_COL_SPACING, mlpColumnXsFromCounts, mlpLayerGap } from "./graph/mlp-layout";

describe("ComputationalGraph", () => {
  const features = { x: true, y: true, xSquared: false, ySquared: false, xTimesY: false, sinX: false, sinY: false };
  const inputIds = constructInputIds(features);

  it("builds resblock with sum node and skip connection", () => {
    const graph = buildResBlockGraph(inputIds, Activations.TANH, Activations.TANH);
    const sumNodes = [...graph.nodes.values()].filter((n) => n.kind === "sum");
    expect(sumNodes.length).toBe(1);
    expect(graph.validate().valid).toBe(true);
  });

  it("builds mini-resnet with two residual blocks", () => {
    const graph = buildMiniResNetGraph(inputIds, Activations.TANH, Activations.TANH);
    const sumNodes = [...graph.nodes.values()].filter((n) => n.kind === "sum");
    expect(sumNodes.length).toBe(2);
  });

  it("forward/backprop runs on graph topology", () => {
    const graph = applyArchitecturePreset("resblock", {
      networkShape: [2],
      numHiddenLayers: 1,
      activation: Activations.TANH,
      outputActivation: Activations.TANH,
      enabledFeatures: features,
    });
    forwardPropGraph(graph, [0.5, -0.3]);
    backPropGraph(graph, 1, Errors.SQUARE);
    updateWeightsGraph(graph, 0.03, 0);
    expect(Number.isFinite(graph.getOutputNode().output)).toBe(true);
  });

  it("rejects cyclic connections", () => {
    const engine = new PlaygroundEngine({ architecturePreset: "mlp", networkShape: [2], numHiddenLayers: 1 });
    const outputId = engine.outputNodeId;
    const hidden = [...engine.graph.nodes.values()].find((n) => n.kind === "dense");
    expect(hidden).toBeDefined();
    const ok = engine.connectNodes(outputId, hidden!.id);
    expect(ok).toBe(false);
  });
});

describe("PlaygroundEngine presets", () => {
  it("trains mini-resnet preset", () => {
    const engine = new PlaygroundEngine({ architecturePreset: "mini-resnet", dataset: "circle" });
    for (let i = 0; i < 20; i++) engine.step();
    expect(engine.epoch).toBe(20);
    expect(engine.boundary[engine.outputNodeId]).toBeDefined();
    expect(Number.isFinite(engine.lossTrain)).toBe(true);
  });
});

describe("mlp layer spacing", () => {
  it("keeps the default gap for small neighboring layers", () => {
    expect(mlpLayerGap(2, 4)).toBe(MLP_COL_SPACING);
    expect(mlpLayerGap(4, 4)).toBe(MLP_COL_SPACING);
    expect(mlpLayerGap(16, 1)).toBe(MLP_COL_SPACING);
  });

  it("widens the gap when two neighboring layers are dense", () => {
    expect(mlpLayerGap(16, 16)).toBeGreaterThan(MLP_COL_SPACING * 2);
    const xs = mlpColumnXsFromCounts([2, 16, 16, 1]);
    expect(xs[2]! - xs[1]!).toBe(mlpLayerGap(16, 16));
    expect(xs[3]! - xs[2]!).toBe(MLP_COL_SPACING);
  });
});
