import { describe, expect, it } from "vitest";
import type { AutogradGraph } from "./graph";
import { AUTOGRAD_PRESETS, buildAutogradPreset, type AutogradPresetId } from "./presets";

// Node box as the playground renders it; mirrors AUTOGRAD_NODE_WIDTH / HEIGHT in the React adapter.
const NODE_WIDTH = 96;
const NODE_HEIGHT = 64;

const [h1, h2, h3] = [0.5 * 1 - 1 * 0.5 + 0.25, -0.5 * 1 + 1 * 0.5 - 0.25, 1 * 1 + 0.5 * 0.5].map(Math.tanh);
const [e1, e2] = [Math.exp(1 * 1), Math.exp(1 * -1)];

/** Closed form of each preset's output at its default leaf values. */
const EXPECTED_OUTPUT: Record<AutogradPresetId, number> = {
  expr: (2 + -3) * 4,
  neuron: Math.tanh(1 * -2 + 0.5 * 1 + 0.5),
  sigmoid: 1 / (1 + Math.exp(-(-2 * 1 + 1 * 0.5 + 0.5))),
  hinge: Math.max(0, 1 - -1 * (1 * 2 + 0.5)),
  twolayer: Math.tanh(-0.5 * Math.tanh(1 * 2 - 1) + 0.25),
  mlp: h1 - h2 + 0.5 * h3 + 0.1,
  attention: (e1 * 2 + e2 * -1) / (e1 + e2),
};

/** Central finite difference of the output with respect to one leaf. */
function numericGrad(graph: AutogradGraph, leafId: string, step = 1e-6): number {
  const leaf = graph.getNode(leafId)!;
  const output = graph.getNode(graph.outputId)!;
  const original = leaf.value;
  leaf.value = original + step;
  graph.forward();
  const plus = output.value;
  leaf.value = original - step;
  graph.forward();
  const minus = output.value;
  leaf.value = original;
  graph.forward();
  return (plus - minus) / (2 * step);
}

function nodeByLabel(graph: AutogradGraph, label: string) {
  return [...graph.nodes.values()].find((n) => n.label === label)!;
}

describe.each<AutogradPresetId>(AUTOGRAD_PRESETS)("preset %s", (preset) => {
  it("computes the documented function", () => {
    const g = buildAutogradPreset(preset);
    g.forward();

    expect(g.getNode(g.outputId)!.value).toBeCloseTo(EXPECTED_OUTPUT[preset], 12);
  });

  it("knows no operation value until a pass is run", () => {
    const g = buildAutogradPreset(preset);

    expect(g.forwardProgress().done).toBe(0);
    expect(g.forwardProgress().total).toBe([...g.nodes.values()].filter((n) => !n.isLeaf).length);
  });

  it("lays out every node without overlaps and keeps every edge left to right", () => {
    const g = buildAutogradPreset(preset);
    const boxes = [...g.nodes.values()].map((n) => {
      const position = g.positions.get(n.id);
      expect(position, `node ${n.label} has no position`).toBeDefined();
      return { x: position!.x, y: position!.y };
    });

    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i];
        const b = boxes[j];
        const overlaps = a.x < b.x + NODE_WIDTH && b.x < a.x + NODE_WIDTH && a.y < b.y + NODE_HEIGHT && b.y < a.y + NODE_HEIGHT;
        expect(overlaps, `nodes ${i} and ${j} overlap`).toBe(false);
      }
    }
    for (const edge of g.edges.values()) {
      const source = g.positions.get(edge.source)!;
      const target = g.positions.get(edge.target)!;
      expect(source.x, `${g.getNode(edge.source)!.label} -> ${g.getNode(edge.target)!.label} goes backwards`).toBeLessThan(
        target.x,
      );
    }
  });

  it("feeds every node into the output, the only sink", () => {
    const g = buildAutogradPreset(preset);

    const sinks = [...g.nodes.values()].filter((n) => g.outputEdges(n.id).length === 0);
    expect(sinks.map((n) => n.id)).toEqual([g.outputId]);

    const upstream = new Set<string>([g.outputId]);
    for (const id of upstream) {
      for (const edge of g.inputEdges(id)) upstream.add(edge.source);
    }
    expect(upstream.size).toBe(g.nodes.size);
  });

  it("matches central finite differences on every leaf", () => {
    const g = buildAutogradPreset(preset);
    g.forward();
    g.backward();

    for (const node of g.nodes.values()) {
      if (!node.isLeaf) continue;
      expect(node.grad, `gradient of ${node.label}`).toBeCloseTo(numericGrad(g, node.id), 5);
    }
  });
});

describe("preset gradients", () => {
  it("sigmoid: the shared constant accumulates the gradient of both its consumers", () => {
    const g = buildAutogradPreset("sigmoid");
    g.forward();
    g.backward();
    const out = g.getNode(g.outputId)!.value;

    // σ'(z) = σ(1 − σ); the constant 1 reaches σ through the numerator and through 1 + e^(−z).
    expect(nodeByLabel(g, "1").grad).toBeCloseTo(out * (1 - out), 12);
  });

  it("hinge: a satisfied margin cuts every gradient at the ReLU kink", () => {
    const g = buildAutogradPreset("hinge");
    // x = −1 gives ŷ = −1.5, so 1 − y·ŷ = −0.5 < 0 and the loss is flat.
    nodeByLabel(g, "x").value = -1;
    g.forward();
    g.backward();

    expect(g.getNode(g.outputId)!.value).toBe(0);
    for (const node of g.nodes.values()) {
      if (node.id === g.outputId) continue;
      expect(node.grad, `gradient of ${node.label}`).toBe(0);
    }
  });

  it("attention: the attention weights sum to one", () => {
    const g = buildAutogradPreset("attention");
    g.forward();

    expect(nodeByLabel(g, "α1").value + nodeByLabel(g, "α2").value).toBeCloseTo(1, 12);
  });
});
