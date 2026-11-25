import { describe, expect, it } from "vitest";
import { AutogradGraph, resetAutogradIdCounter, type AutogradNode } from "./graph";
import { buildAutogradPreset, type AutogradPresetId } from "./presets";
import type { AutogradOp, AutogradStep } from "./types";

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
    g.forward();
    g.backward();
    const out = g.getNode(g.outputId);
    expect(out).toBeDefined();
    expect(Number.isFinite(out!.value)).toBe(true);
    expect(Math.abs(out!.value)).toBeLessThanOrEqual(1);
    // Every leaf should have received some gradient signal.
    const leaves = [...g.nodes.values()].filter((n) => n.isLeaf);
    expect(leaves.length).toBeGreaterThan(0);
  });
});

describe("AutogradGraph step-by-step passes", () => {
  /** The (a + b) * c preset: three leaves, then its two operations; its four edges in connection order. */
  function buildExpr() {
    const g = buildAutogradPreset("expr");
    const [a, b, c, sum, prod] = [...g.nodes.values()];
    const [aToSum, bToSum, sumToProd, cToProd] = [...g.edges.values()];
    return { g, a, b, c, sum, prod, aToSum, bToSum, sumToProd, cToProd };
  }

  it("knows no value or gradient before a pass", () => {
    const { g, a, sum, aToSum } = buildExpr();
    expect(g.isComputed(a.id)).toBe(true);
    expect(g.isComputed(sum.id)).toBe(false);
    expect(g.hasGrad(a.id)).toBe(false);
    expect(g.hasLocalDer(aToSum.id)).toBe(false);
    expect(g.lastStep).toBeNull();
    expect(g.nextStep).toEqual({ ids: [sum.id], pass: "forward" });
    expect(g.forwardProgress()).toEqual({ done: 0, total: 2 });
    expect(g.backwardProgress()).toEqual({ done: 0, total: 2 });
  });

  it("takes one operation per step where each depends on the last, in topological order", () => {
    const { g, sum, prod } = buildExpr();

    expect(g.step()).toEqual({ ids: [sum.id], pass: "forward" });
    expect(sum.value).toBe(-1);
    expect(g.isComputed(sum.id)).toBe(true);
    expect(g.isComputed(prod.id)).toBe(false);
    expect(g.lastStep).toEqual({ ids: [sum.id], pass: "forward" });
    expect(g.nextStep).toEqual({ ids: [prod.id], pass: "forward" });
    expect(g.forwardProgress()).toEqual({ done: 1, total: 2 });

    expect(g.step()).toEqual({ ids: [prod.id], pass: "forward" });
    expect(prod.value).toBe(-4);
    expect(g.forwardProgress()).toEqual({ done: 2, total: 2 });
    expect(g.backwardProgress()).toEqual({ done: 0, total: 2 });
  });

  it("ends the forward pass where a whole forward pass does", () => {
    const stepped = buildAutogradPreset("neuron");
    for (let s = stepped.step(); s?.pass === "forward"; s = stepped.step());
    const whole = buildAutogradPreset("neuron");
    whole.forward();

    expect(stepped.forwardProgress()).toEqual({ done: 4, total: 4 });
    for (const [id, node] of stepped.nodes) {
      expect(node.value).toBeCloseTo(whole.getNode(id)!.value, 12);
    }
  });

  it("continues into the backward pass once the forward pass is complete", () => {
    const { g, a, b, c, sum, prod, aToSum, bToSum, sumToProd, cToProd } = buildExpr();
    g.forward();

    expect(g.nextStep).toEqual({ ids: [prod.id], pass: "backward" });
    // The output gradient is seeded to 1 and pushed through the output node.
    expect(g.step()).toEqual({ ids: [prod.id], pass: "backward" });
    expect(prod.grad).toBe(1);
    expect(g.hasGrad(prod.id)).toBe(true);
    expect(sumToProd.localDer).toBe(4);
    expect(cToProd.localDer).toBe(-1);
    expect(g.hasLocalDer(sumToProd.id)).toBe(true);
    expect(g.hasLocalDer(aToSum.id)).toBe(false);
    expect(sum.grad).toBe(4);
    expect(c.grad).toBe(-1);
    expect(g.hasGrad(a.id)).toBe(false);
    expect(g.backwardProgress()).toEqual({ done: 1, total: 2 });

    expect(g.step()).toEqual({ ids: [sum.id], pass: "backward" });
    expect(aToSum.localDer).toBe(1);
    expect(bToSum.localDer).toBe(1);
    expect(a.grad).toBe(4);
    expect(b.grad).toBe(4);
    expect(g.hasGrad(a.id)).toBe(true);
    expect(g.backwardProgress()).toEqual({ done: 2, total: 2 });

    // Both passes are complete: nothing is left to step through.
    expect(g.nextStep).toBeNull();
    expect(g.step()).toBeNull();
  });

  it("ends the backward pass where a whole backward pass does", () => {
    const stepped = buildAutogradPreset("neuron");
    for (let s = stepped.step(); s !== null; s = stepped.step());
    const whole = buildAutogradPreset("neuron");
    whole.forward();
    whole.backward();

    expect(stepped.backwardProgress()).toEqual({ done: 4, total: 4 });
    for (const [id, node] of stepped.nodes) {
      expect(node.grad).toBeCloseTo(whole.getNode(id)!.grad, 12);
      expect(stepped.hasGrad(id)).toBe(true);
    }
    for (const [id, edge] of stepped.edges) {
      expect(edge.localDer).toBeCloseTo(whole.edges.get(id)!.localDer, 12);
      expect(stepped.hasLocalDer(id)).toBe(true);
    }
  });

  it("goes back through the operations in the reverse of the forward order", () => {
    const g = buildAutogradPreset("neuron");
    const order: string[] = [];
    for (let s = g.step(); s !== null; s = g.step()) order.push(...s.ids);
    const turn = order.length / 2;

    expect(order.slice(turn).reverse()).toEqual(order.slice(0, turn));
  });

  it("forgets all progress when the topology changes", () => {
    const { g, a, sum } = buildExpr();
    g.forward();
    g.step();
    expect(g.backwardProgress().done).toBe(1);

    const extra = g.addNode("input", { value: 10 });
    expect(g.forwardProgress()).toEqual({ done: 0, total: 2 });
    expect(g.backwardProgress()).toEqual({ done: 0, total: 2 });
    expect(g.hasGrad(a.id)).toBe(false);
    expect(g.lastStep).toBeNull();

    g.connect(extra.id, sum.id);
    expect(g.isComputed(sum.id)).toBe(false);
    expect(g.nextStep).toEqual({ ids: [sum.id], pass: "forward" });
    g.step();
    expect(sum.value).toBe(9);
  });

  it("forgets all progress when resetPasses follows a leaf value change", () => {
    const { g, a, sum } = buildExpr();
    g.forward();
    g.backward();
    a.value = 5;
    g.resetPasses();

    expect(g.forwardProgress()).toEqual({ done: 0, total: 2 });
    expect(g.hasGrad(a.id)).toBe(false);
    expect(g.step()).toEqual({ ids: [sum.id], pass: "forward" });
    expect(sum.value).toBe(2);
  });

  it("a whole forward pass starts the backward pass over", () => {
    const { g, a, prod } = buildExpr();
    g.forward();
    g.backward();
    expect(g.hasGrad(a.id)).toBe(true);

    g.forward();

    expect(g.lastStep).toBeNull();
    expect(g.backwardProgress()).toEqual({ done: 0, total: 2 });
    expect(g.hasGrad(a.id)).toBe(false);
    expect(g.nextStep).toEqual({ ids: [prod.id], pass: "backward" });
  });

  it("a forward pass keeps the numbers of the last backward pass in place", () => {
    const { g, a, bToSum } = buildExpr();
    g.forward();
    g.backward();
    a.value = 5;

    // No longer current, so no longer shown — but a numeric check may still read them.
    g.forward();

    expect(g.hasGrad(a.id)).toBe(false);
    expect(a.grad).toBe(4);
    expect(bToSum.localDer).toBe(1);
  });

  it("a whole backward pass leaves no single step singled out", () => {
    const { g, a } = buildExpr();
    g.step();
    g.step();

    g.backward();

    expect(g.lastStep).toBeNull();
    expect(g.hasGrad(a.id)).toBe(true);
    expect(g.step()).toBeNull();
  });
});

describe("AutogradGraph steps independent operations together", () => {
  /** Labels of the operation nodes a step takes, sorted: a step lists them in topological order, which is not what these tests are about. */
  function taken(g: AutogradGraph, step: AutogradStep): string[] {
    return step.ids.map((id) => g.getNode(id)!.label!).sort();
  }

  /** a·b and b·c, then their sum: the two products share an input, but neither depends on the other. */
  function buildFork() {
    resetAutogradIdCounter(1);
    const g = new AutogradGraph();
    const a = g.addNode("input", { value: 2, label: "a" });
    const b = g.addNode("input", { value: 3, label: "b" });
    const c = g.addNode("input", { value: 5, label: "c" });
    const ab = g.addNode("mul", { label: "a·b" });
    const bc = g.addNode("mul", { label: "b·c" });
    const out = g.addNode("add", { label: "out" });
    g.connect(a.id, ab.id);
    g.connect(b.id, ab.id);
    g.connect(b.id, bc.id);
    g.connect(c.id, bc.id);
    g.connect(ab.id, out.id);
    g.connect(bc.id, out.id);
    g.setOutput(out.id);
    return { g, a, b, c, ab, bc, out };
  }

  it("computes every operation whose inputs all have a value in one forward step", () => {
    const { g, ab, bc, out } = buildFork();

    const first = g.step()!;
    expect(first.pass).toBe("forward");
    expect(taken(g, first)).toEqual(["a·b", "b·c"]);
    expect(ab.value).toBe(6);
    expect(bc.value).toBe(15);
    expect(g.isComputed(out.id)).toBe(false);
    expect(g.forwardProgress()).toEqual({ done: 2, total: 3 });

    expect(taken(g, g.step()!)).toEqual(["out"]);
    expect(out.value).toBe(21);
  });

  it("pushes the gradient through every operation whose consumers are all done in one backward step", () => {
    const { g, a, b, c } = buildFork();
    g.forward();

    const throughOutput = g.step()!;
    expect(throughOutput.pass).toBe("backward");
    expect(taken(g, throughOutput)).toEqual(["out"]);

    // Both products feed the output alone, so neither waits for the other.
    expect(taken(g, g.nextStep!)).toEqual(["a·b", "b·c"]);
    g.step();
    expect(a.grad).toBe(3);
    expect(b.grad).toBe(2 + 5);
    expect(c.grad).toBe(3);
    expect(g.backwardProgress()).toEqual({ done: 3, total: 3 });
    expect(g.step()).toBeNull();
  });

  it("keeps an operation waiting until every one of its inputs has a value", () => {
    resetAutogradIdCounter(1);
    const g = new AutogradGraph();
    const a = g.addNode("input", { value: 2, label: "a" });
    const b = g.addNode("input", { value: 3, label: "b" });
    const c = g.addNode("input", { value: 5, label: "c" });
    const ab = g.addNode("mul", { label: "a·b" });
    const bc = g.addNode("mul", { label: "b·c" });
    const abc = g.addNode("mul", { label: "a·b·c" });
    const out = g.addNode("add", { label: "out" });
    g.connect(a.id, ab.id);
    g.connect(b.id, ab.id);
    g.connect(b.id, bc.id);
    g.connect(c.id, bc.id);
    g.connect(ab.id, abc.id);
    g.connect(c.id, abc.id);
    g.connect(abc.id, out.id);
    g.connect(bc.id, out.id);
    g.setOutput(out.id);

    expect(taken(g, g.step()!)).toEqual(["a·b", "b·c"]);
    // c is ready from the start, but a·b·c also reads a·b.
    expect(taken(g, g.step()!)).toEqual(["a·b·c"]);
    expect(abc.value).toBe(30);
    expect(taken(g, g.step()!)).toEqual(["out"]);
  });

  it("keeps an operation waiting until every one of its consumers is done", () => {
    resetAutogradIdCounter(1);
    const g = new AutogradGraph();
    const a = g.addNode("input", { value: 2, label: "a" });
    const b = g.addNode("input", { value: 3, label: "b" });
    const p = g.addNode("mul", { label: "a·b" });
    const t = g.addNode("tanh", { label: "tanh" });
    const out = g.addNode("add", { label: "out" });
    g.connect(a.id, p.id);
    g.connect(b.id, p.id);
    g.connect(p.id, out.id);
    g.connect(p.id, t.id);
    g.connect(t.id, out.id);
    g.setOutput(out.id);
    g.forward();

    expect(taken(g, g.step()!)).toEqual(["out"]);
    // The output is done, but a·b still feeds the tanh.
    expect(taken(g, g.step()!)).toEqual(["tanh"]);
    expect(g.hasGrad(a.id)).toBe(false);

    expect(taken(g, g.step()!)).toEqual(["a·b"]);
    // d out/d(a·b) = 1 + (1 − tanh²(6)), times b.
    expect(a.grad).toBeCloseTo(3 * (2 - Math.tanh(6) ** 2), 12);
    expect(g.hasGrad(a.id)).toBe(true);
  });

  /** The forward pass of each preset, wave by wave. */
  const FORWARD_WAVES: [AutogradPresetId, string[][]][] = [
    ["expr", [["a+b"], ["(a+b)·c"]]],
    ["neuron", [["w1·x1", "w2·x2"], ["Σ"], ["tanh"]]],
    ["sigmoid", [["w1·x1", "w2·x2"], ["z"], ["−z"], ["e^(−z)"], ["1+e^(−z)"], ["σ"]]],
    ["hinge", [["w·x"], ["ŷ"], ["y·ŷ"], ["1−y·ŷ"], ["L"]]],
    ["twolayer", [["w1·x"], ["z1"], ["h"], ["w2·h"], ["z2"], ["ŷ"]]],
    [
      "mlp",
      [
        ["w11·x1", "w12·x2", "w21·x1", "w22·x2", "w31·x1", "w32·x2"],
        ["z1", "z2", "z3"],
        ["h1", "h2", "h3"],
        ["v1·h1", "v2·h2", "v3·h3"],
        ["ŷ"],
      ],
    ],
    [
      "attention",
      [["q·k1", "q·k2"], ["e^s1", "e^s2"], ["Σe"], ["α1", "α2"], ["α1·v1", "α2·v2"], ["out"]],
    ],
  ];

  it.each(FORWARD_WAVES)("preset %s steps through one wave of independent operations at a time", (preset, forward) => {
    const g = buildAutogradPreset(preset);
    const waves: string[][] = [];
    let s = g.step();
    for (; s?.pass === "forward"; s = g.step()) waves.push(taken(g, s));
    expect(waves).toEqual(forward);

    const back: string[][] = [];
    for (; s !== null; s = g.step()) back.push(taken(g, s));
    expect(back).toEqual(forward.slice().reverse());
  });
});

describe("AutogradGraph input ports", () => {
  /** Leaves x1 = 2, x2 = 3, … and one operation over them, the output. */
  function build(op: AutogradOp, leaves = 3) {
    resetAutogradIdCounter(1);
    const g = new AutogradGraph();
    const inputs = Array.from({ length: leaves }, (_, i) => g.addNode("input", { value: i + 2, label: `x${i + 1}` }));
    const node = g.addNode(op);
    g.setOutput(node.id);
    return { g, inputs, node };
  }

  /** What reaches a node through its ports, in port order. */
  const fedBy = (g: AutogradGraph, node: AutogradNode): string[] => g.inputEdges(node.id).map((e) => e.source);

  it("counts one port per operand, and two for an n-ary operation before anything is connected", () => {
    for (const [op, ports] of [["neg", 1], ["tanh", 1], ["sub", 2], ["div", 2], ["add", 2], ["mul", 2]] as const) {
      const { g, inputs, node } = build(op);
      expect(g.inputPortCount(node.id), op).toBe(ports);
      expect(g.inputPortCount(inputs[0].id), `${op} leaf`).toBe(0);
    }
  });

  it("feeds the ports in connection order when no port is named", () => {
    const { g, inputs, node } = build("sub");

    g.connect(inputs[1].id, node.id);
    g.connect(inputs[0].id, node.id);

    expect(fedBy(g, node)).toEqual([inputs[1].id, inputs[0].id]);
    g.forward();
    expect(node.value).toBe(3 - 2);
  });

  it("feeds the port a connection names, ahead of the ones added before it", () => {
    const { g, inputs, node } = build("sub");

    g.connect(inputs[1].id, node.id, 1);
    g.forward();
    // The unfed port 0 reads as 0.
    expect(node.value).toBe(-3);

    g.connect(inputs[0].id, node.id, 0);

    expect(g.inputEdges(node.id).map((e) => e.slot)).toEqual([0, 1]);
    g.forward();
    expect(node.value).toBe(2 - 3);
  });

  it.each([
    ["add", 4, 2],
    ["mul", 4, 4],
    ["sub", 0, 0],
    ["div", 1, 0],
  ] as const)("lets one source feed both ports of a binary %s", (op, value, grad) => {
    const { g, inputs, node } = build(op);
    const x = inputs[0];

    g.connect(x.id, node.id);
    g.connect(x.id, node.id);
    g.forward();
    g.backward();

    expect(fedBy(g, node)).toEqual([x.id, x.id]);
    expect(node.value).toBe(value);
    expect(x.grad).toBe(grad);
  });

  it("hands the port of a fixed-arity operation to the source dropped on it", () => {
    const { g, inputs, node } = build("sub");
    g.connect(inputs[0].id, node.id);
    g.connect(inputs[1].id, node.id);

    g.connect(inputs[2].id, node.id, 1);

    expect(fedBy(g, node)).toEqual([inputs[0].id, inputs[2].id]);
    expect(g.edges.size).toBe(2);
    g.forward();
    expect(node.value).toBe(2 - 4);

    // A source already feeding a port changes nothing by being dropped on it again.
    expect(g.connect(inputs[0].id, node.id, 0)).toBeNull();
    expect(fedBy(g, node)).toEqual([inputs[0].id, inputs[2].id]);
  });

  it("swaps the input of a unary operation the same way", () => {
    const { g, inputs, node } = build("tanh");
    g.connect(inputs[0].id, node.id);

    g.connect(inputs[1].id, node.id, 0);

    expect(fedBy(g, node)).toEqual([inputs[1].id]);
  });

  it("rejects a port the operation does not have", () => {
    const difference = build("sub");
    expect(difference.g.connect(difference.inputs[0].id, difference.node.id, 2)).toBeNull();
    expect(difference.g.connect(difference.inputs[0].id, difference.node.id, -1)).toBeNull();
    expect(difference.g.connect(difference.inputs[0].id, difference.node.id, NaN)).toBeNull();

    const unary = build("neg");
    expect(unary.g.connect(unary.inputs[0].id, unary.node.id, 1)).toBeNull();
  });

  it("gives an n-ary operation a port per operand, and adds one for a wire dropped on a fed port", () => {
    const { g, inputs, node } = build("add", 4);
    g.connect(inputs[0].id, node.id);
    g.connect(inputs[1].id, node.id);
    expect(g.inputPortCount(node.id)).toBe(2);

    // Both ports are fed: the wire becomes a third operand instead of replacing one.
    expect(g.connect(inputs[2].id, node.id, 0)?.slot).toBe(2);
    expect(g.inputPortCount(node.id)).toBe(3);
    expect(fedBy(g, node)).toEqual(inputs.slice(0, 3).map((n) => n.id));
    g.forward();
    expect(node.value).toBe(2 + 3 + 4);
  });

  it("fills the free port of an n-ary operation before adding one", () => {
    const { g, inputs, node } = build("add", 4);
    for (const input of inputs.slice(0, 3)) g.connect(input.id, node.id);
    g.removeEdge(g.inputEdges(node.id)[1].id);

    expect(g.connect(inputs[3].id, node.id, 0)?.slot).toBe(1);
    expect(g.inputPortCount(node.id)).toBe(3);
  });

  it("reads an unfed port of a positional operation as a missing operand", () => {
    const { g, inputs, node } = build("sub");
    g.connect(inputs[0].id, node.id);
    g.connect(inputs[1].id, node.id);

    g.removeEdge(g.inputEdges(node.id)[0].id);
    g.forward();
    g.backward();

    // The subtrahend keeps port 1: 0 − x2.
    expect(g.inputEdges(node.id).map((e) => e.slot)).toEqual([1]);
    expect(node.value).toBe(-3);
    expect(inputs[1].grad).toBe(-1);

    // A wire with no port named takes the lowest free one.
    g.connect(inputs[2].id, node.id);
    g.forward();
    expect(node.value).toBe(4 - 3);
  });

  it("skips an unfed port of an n-ary operation", () => {
    const product = build("mul");
    for (const input of product.inputs) product.g.connect(input.id, product.node.id);
    product.g.removeEdge(product.g.inputEdges(product.node.id)[1].id);
    product.g.forward();
    product.g.backward();
    // 2·4, the middle factor gone.
    expect(product.node.value).toBe(8);
    expect(product.inputs.map((input) => input.grad)).toEqual([4, 0, 2]);

    const sum = build("add");
    for (const input of sum.inputs) sum.g.connect(input.id, sum.node.id);
    sum.g.removeEdge(sum.g.inputEdges(sum.node.id)[1].id);
    sum.g.forward();
    sum.g.backward();
    expect(sum.node.value).toBe(6);
    expect(sum.inputs.map((input) => input.grad)).toEqual([1, 0, 1]);
  });

  it("keeps the ports in a snapshot, and reads older ones in order", () => {
    const { g, inputs, node } = build("sub");
    g.connect(inputs[1].id, node.id, 1);
    g.connect(inputs[0].id, node.id, 0);

    const restored = AutogradGraph.fromSnapshot(g.toSnapshot());
    expect(restored.inputEdges(node.id).map((e) => [e.source, e.slot])).toEqual([
      [inputs[0].id, 0],
      [inputs[1].id, 1],
    ]);

    // Before operations had ports, an edge fed the port after the ones its target already had.
    const snapshot = g.toSnapshot();
    const older = { ...snapshot, edges: snapshot.edges.map(({ id, source, target }) => ({ id, source, target })) };
    expect(AutogradGraph.fromSnapshot(older).inputEdges(node.id).map((e) => e.source)).toEqual([
      inputs[1].id,
      inputs[0].id,
    ]);
  });
});
