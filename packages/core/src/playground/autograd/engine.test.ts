import { describe, expect, it } from "vitest";
import { CompGraphEngine } from "./engine";

/** Operation node ids of the default (a + b) * c preset, in step order. */
function opIds(engine: CompGraphEngine): string[] {
  return [...engine.graph.nodes.values()].filter((n) => !n.isLeaf).map((n) => n.id);
}

function leaf(engine: CompGraphEngine, label: string) {
  return [...engine.graph.nodes.values()].find((n) => n.label === label)!;
}

function expectNothingComputed(engine: CompGraphEngine, total = 2): void {
  expect(engine.graph.forwardProgress()).toEqual({ done: 0, total });
  expect(engine.graph.backwardProgress()).toEqual({ done: 0, total });
  expect(engine.graph.lastStep).toBeNull();
}

describe("CompGraphEngine", () => {
  it("knows no value or gradient until a pass is run", () => {
    const engine = new CompGraphEngine();
    expectNothingComputed(engine);
    expect(engine.graph.hasGrad(leaf(engine, "a").id)).toBe(false);
  });

  it("runStep goes through the forward pass one operation node at a time", () => {
    const engine = new CompGraphEngine();
    const [sum, prod] = opIds(engine);

    expect(engine.runStep()).toEqual({ id: sum, pass: "forward" });
    expect(engine.graph.isComputed(sum)).toBe(true);
    expect(engine.graph.isComputed(prod)).toBe(false);
    expect(engine.runStep()).toEqual({ id: prod, pass: "forward" });
  });

  it("runStep continues into the backward pass once the forward pass is complete", () => {
    const engine = new CompGraphEngine();
    const a = leaf(engine, "a");
    const [sum, prod] = opIds(engine);

    engine.runForward();
    expect(engine.graph.hasGrad(a.id)).toBe(false);

    expect(engine.runStep()).toEqual({ id: prod, pass: "backward" });
    expect(engine.runStep()).toEqual({ id: sum, pass: "backward" });
    expect(a.grad).toBe(4);
    expect(engine.runStep()).toBeNull();
  });

  it("runForward completes a partly stepped pass", () => {
    const engine = new CompGraphEngine();
    engine.runStep();
    engine.runForward();

    expect(engine.graph.forwardProgress()).toEqual({ done: 2, total: 2 });
    expect(engine.graph.lastStep).toBeNull();
  });

  it("runBackward computes every value and gradient at once", () => {
    const engine = new CompGraphEngine();
    const a = leaf(engine, "a");
    engine.runStep();
    expect(a.grad).toBe(0);
    expect(engine.graph.hasGrad(a.id)).toBe(false);

    engine.runBackward();

    expect(engine.graph.forwardProgress()).toEqual({ done: 2, total: 2 });
    expect(engine.graph.backwardProgress()).toEqual({ done: 2, total: 2 });
    expect(a.grad).toBe(4);
    expect(engine.graph.hasGrad(a.id)).toBe(true);
  });

  it("a leaf value edit forgets the values and gradients computed so far", () => {
    const engine = new CompGraphEngine();
    const a = leaf(engine, "a");
    const [sum] = opIds(engine);
    engine.runForward();
    engine.runStep();

    engine.setNodeValue(a.id, 7);

    expectNothingComputed(engine);
    expect(engine.runStep()).toEqual({ id: sum, pass: "forward" });
    expect(engine.graph.getNode(sum)!.value).toBe(4);
  });

  it("a topology edit forgets the values and gradients computed so far", () => {
    const engine = new CompGraphEngine();
    const [sum] = opIds(engine);
    engine.runBackward();

    const extra = engine.addPaletteNode("input", { x: 0, y: 0 });
    engine.setNodeValue(extra, 10);
    engine.connectNodes(extra, sum);

    expectNothingComputed(engine);
    engine.runStep();
    expect(engine.graph.getNode(sum)!.value).toBe(9);
  });

  it("reset and loadPreset hand back a graph with nothing computed", () => {
    const engine = new CompGraphEngine();
    engine.runBackward();
    engine.reset();
    expectNothingComputed(engine);

    engine.runStep();
    engine.loadPreset("neuron");
    expectNothingComputed(engine, 4);
  });
});
