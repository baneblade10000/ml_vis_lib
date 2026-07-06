import { describe, expect, it } from "vitest";
import { AutogradGraph, resetAutogradIdCounter } from "./graph";
import { buildAutogradPreset } from "./presets";

describe("AutogradGraph", () => {
  it("forward + backward on (a+b)*c matches analytic gradients", () => {
    resetAutogradIdCounter(1);
    const g = new AutogradGraph();
    const a = g.addNode("input", { value: 2 });
    const b = g.addNode("input", { value: -3 });
    const c = g.addNode("input", { value: 4 });
    const sum = g.addNode("add");
    const prod = g.addNode("mul");
    g.connect(a.id, sum.id);
    g.connect(b.id, sum.id);
    g.connect(sum.id, prod.id);
    g.connect(c.id, prod.id);
    g.setOutput(prod.id);

    g.forward();
    // (2 + -3) * 4 = -4
    expect(prod.value).toBe(-4);

    g.backward();
    // out = (a+b)*c => d/da = c, d/db = c, d/dc = a+b, d/dsum = c
    expect(a.grad).toBe(4);
    expect(b.grad).toBe(4);
    expect(c.grad).toBe(-1);
    expect(sum.grad).toBe(4);
    expect(prod.grad).toBe(1);
  });

  it("computes tanh derivative correctly", () => {
    resetAutogradIdCounter(1);
    const g = new AutogradGraph();
    const x = g.addNode("input", { value: 0.5 });
    const t = g.addNode("tanh");
    g.connect(x.id, t.id);
    g.setOutput(t.id);
    g.forward();
    g.backward();
    const expected = 1 - Math.tanh(0.5) ** 2;
    expect(x.grad).toBeCloseTo(expected, 10);
  });

  it("respects input order for div", () => {
    resetAutogradIdCounter(1);
    const g = new AutogradGraph();
    const a = g.addNode("input", { value: 6 });
    const b = g.addNode("input", { value: 2 });
    const d = g.addNode("div");
    g.connect(a.id, d.id);
    g.connect(b.id, d.id);
    g.setOutput(d.id);
    g.forward();
    g.backward();
    expect(d.value).toBe(3);
    // d/da = 1/b = 0.5 ; d/db = -a/b^2 = -1.5
    expect(a.grad).toBeCloseTo(0.5, 10);
    expect(b.grad).toBeCloseTo(-1.5, 10);
  });

  it("rejects cycles and over-arity connections", () => {
    resetAutogradIdCounter(1);
    const g = new AutogradGraph();
    const a = g.addNode("input", { value: 1 });
    const b = g.addNode("input", { value: 1 });
    const sub = g.addNode("sub");
    expect(g.connect(a.id, sub.id)).not.toBeNull();
    expect(g.connect(b.id, sub.id)).not.toBeNull();
    // sub arity is 2; a third input must be rejected.
    const c = g.addNode("input", { value: 1 });
    expect(g.connect(c.id, sub.id)).toBeNull();
    // connecting sub back into a leaf is rejected.
    expect(g.connect(sub.id, a.id)).toBeNull();
  });

  it("neuron preset produces a finite output and gradients", () => {
    const g = buildAutogradPreset("neuron");
    const out = g.getNode(g.outputId);
    expect(out).toBeDefined();
    expect(Number.isFinite(out!.value)).toBe(true);
    expect(Math.abs(out!.value)).toBeLessThanOrEqual(1);
    // Every leaf should have received some gradient signal.
    const leaves = [...g.nodes.values()].filter((n) => n.isLeaf);
    expect(leaves.length).toBeGreaterThan(0);
  });
});
