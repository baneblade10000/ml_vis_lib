import { AutogradGraph, resetAutogradIdCounter } from "./graph";
import type { AutogradPosition } from "./types";

export type AutogradPresetId =
  | "expr"
  | "neuron"
  | "sigmoid"
  | "hinge"
  | "twolayer"
  | "mlp"
  | "attention";

const COL = 200;
const ROW = 110;

/** (a + b) * c — the classic intro example for computational graphs. */
function buildExpr(): AutogradGraph {
  resetAutogradIdCounter(1);
  const graph = new AutogradGraph();
  const a = graph.addNode("input", { value: 2, label: "a", position: { x: 0, y: 0 } });
  const b = graph.addNode("input", { value: -3, label: "b", position: { x: 0, y: ROW } });
  const c = graph.addNode("input", { value: 4, label: "c", position: { x: 0, y: 2 * ROW } });
  const sum = graph.addNode("add", { label: "a+b", position: { x: COL, y: ROW / 2 } });
  const prod = graph.addNode("mul", { label: "(a+b)·c", position: { x: 2 * COL, y: ROW } });

  graph.connect(a.id, sum.id);
  graph.connect(b.id, sum.id);
  graph.connect(sum.id, prod.id);
  graph.connect(c.id, prod.id);
  graph.setOutput(prod.id);
  return graph;
}

/** tanh(w1*x1 + w2*x2 + b) — a single neuron, bridging to neural networks. */
function buildNeuron(): AutogradGraph {
  resetAutogradIdCounter(1);
  const graph = new AutogradGraph();
  const x1 = graph.addNode("input", { value: 1, label: "x1", position: { x: 0, y: 0 } });
  const w1 = graph.addNode("input", { value: -2, label: "w1", position: { x: 0, y: ROW } });
  const x2 = graph.addNode("input", { value: 0.5, label: "x2", position: { x: 0, y: 2 * ROW } });
  const w2 = graph.addNode("input", { value: 1, label: "w2", position: { x: 0, y: 3 * ROW } });
  const bias = graph.addNode("input", { value: 0.5, label: "b", position: { x: 0, y: 4 * ROW } });

  const m1 = graph.addNode("mul", { label: "w1·x1", position: { x: COL, y: ROW / 2 } });
  const m2 = graph.addNode("mul", { label: "w2·x2", position: { x: COL, y: 2.5 * ROW } });
  const sum = graph.addNode("add", { label: "Σ", position: { x: 2 * COL, y: 2 * ROW } });
  const out = graph.addNode("tanh", { label: "tanh", position: { x: 3 * COL, y: 2 * ROW } });

  graph.connect(x1.id, m1.id);
  graph.connect(w1.id, m1.id);
  graph.connect(x2.id, m2.id);
  graph.connect(w2.id, m2.id);
  graph.connect(m1.id, sum.id);
  graph.connect(m2.id, sum.id);
  graph.connect(bias.id, sum.id);
  graph.connect(sum.id, out.id);
  graph.setOutput(out.id);
  return graph;
}

/**
 * σ(z) = 1 / (1 + e^(−z)) with z = w1·x1 + w2·x2 + b — the "neuron" preset with
 * its activation spelled out. The constant 1 feeds two consumers, so its
 * gradient sums over both paths.
 */
function buildSigmoid(): AutogradGraph {
  resetAutogradIdCounter(1);
  const graph = new AutogradGraph();
  const x1 = graph.addNode("input", { value: 1, label: "x1", position: { x: 0, y: 0 } });
  const w1 = graph.addNode("input", { value: -2, label: "w1", position: { x: 0, y: ROW } });
  const x2 = graph.addNode("input", { value: 0.5, label: "x2", position: { x: 0, y: 2 * ROW } });
  const w2 = graph.addNode("input", { value: 1, label: "w2", position: { x: 0, y: 3 * ROW } });
  const bias = graph.addNode("input", { value: 0.5, label: "b", position: { x: COL, y: 3.5 * ROW } });

  const m1 = graph.addNode("mul", { label: "w1·x1", position: { x: COL, y: ROW / 2 } });
  const m2 = graph.addNode("mul", { label: "w2·x2", position: { x: COL, y: 2.5 * ROW } });
  const z = graph.addNode("add", { label: "z", position: { x: 2 * COL, y: 2.5 * ROW } });
  const neg = graph.addNode("neg", { label: "−z", position: { x: 3 * COL, y: 2.5 * ROW } });
  const exp = graph.addNode("exp", { label: "e^(−z)", position: { x: 4 * COL, y: 2.5 * ROW } });
  const one = graph.addNode("const", { value: 1, label: "1", position: { x: 4 * COL, y: ROW / 2 } });
  const denom = graph.addNode("add", { label: "1+e^(−z)", position: { x: 5 * COL, y: 2.5 * ROW } });
  const out = graph.addNode("div", { label: "σ", position: { x: 6 * COL, y: 2.5 * ROW } });

  graph.connect(x1.id, m1.id);
  graph.connect(w1.id, m1.id);
  graph.connect(x2.id, m2.id);
  graph.connect(w2.id, m2.id);
  graph.connect(m1.id, z.id);
  graph.connect(m2.id, z.id);
  graph.connect(bias.id, z.id);
  graph.connect(z.id, neg.id);
  graph.connect(neg.id, exp.id);
  graph.connect(exp.id, denom.id);
  graph.connect(one.id, denom.id);
  // div is positional: the numerator goes in first.
  graph.connect(one.id, out.id);
  graph.connect(denom.id, out.id);
  graph.setOutput(out.id);
  return graph;
}

/**
 * L = max(0, 1 − y·ŷ) with ŷ = w·x + b — the SVM hinge loss for a target
 * y ∈ {−1, +1}. Once the margin is satisfied the ReLU kink cuts every
 * gradient to zero.
 */
function buildHinge(): AutogradGraph {
  resetAutogradIdCounter(1);
  const graph = new AutogradGraph();
  const x = graph.addNode("input", { value: 1, label: "x", position: { x: 0, y: 0 } });
  const w = graph.addNode("input", { value: 2, label: "w", position: { x: 0, y: ROW } });
  const bias = graph.addNode("input", { value: 0.5, label: "b", position: { x: COL, y: 2 * ROW } });
  const y = graph.addNode("input", { value: -1, label: "y", position: { x: 2 * COL, y: 3 * ROW } });
  const one = graph.addNode("const", { value: 1, label: "1", position: { x: 3 * COL, y: 4 * ROW } });

  const wx = graph.addNode("mul", { label: "w·x", position: { x: COL, y: ROW / 2 } });
  const yhat = graph.addNode("add", { label: "ŷ", position: { x: 2 * COL, y: ROW } });
  const prod = graph.addNode("mul", { label: "y·ŷ", position: { x: 3 * COL, y: 1.5 * ROW } });
  const margin = graph.addNode("sub", { label: "1−y·ŷ", position: { x: 4 * COL, y: 2 * ROW } });
  const loss = graph.addNode("relu", { label: "L", position: { x: 5 * COL, y: 2.5 * ROW } });

  graph.connect(x.id, wx.id);
  graph.connect(w.id, wx.id);
  graph.connect(wx.id, yhat.id);
  graph.connect(bias.id, yhat.id);
  graph.connect(y.id, prod.id);
  graph.connect(yhat.id, prod.id);
  // sub is positional: the minuend (the constant 1) goes in first.
  graph.connect(one.id, margin.id);
  graph.connect(prod.id, margin.id);
  graph.connect(margin.id, loss.id);
  graph.setOutput(loss.id);
  return graph;
}

/**
 * ŷ = tanh(w2·tanh(w1·x + b1) + b2) — two stacked tanh neurons. The output
 * gradient has to thread two saturating nonlinearities before it reaches
 * x, w1 and b1.
 */
function buildTwoLayer(): AutogradGraph {
  resetAutogradIdCounter(1);
  const graph = new AutogradGraph();
  const x = graph.addNode("input", { value: 1, label: "x", position: { x: 0, y: 0 } });
  const w1 = graph.addNode("input", { value: 2, label: "w1", position: { x: 0, y: ROW } });
  const b1 = graph.addNode("input", { value: -1, label: "b1", position: { x: COL, y: 2 * ROW } });
  const w2 = graph.addNode("input", { value: -0.5, label: "w2", position: { x: 3 * COL, y: 3 * ROW } });
  const b2 = graph.addNode("input", { value: 0.25, label: "b2", position: { x: 4 * COL, y: 4 * ROW } });

  const m1 = graph.addNode("mul", { label: "w1·x", position: { x: COL, y: ROW / 2 } });
  const pre1 = graph.addNode("add", { label: "z1", position: { x: 2 * COL, y: ROW } });
  const hidden = graph.addNode("tanh", { label: "h", position: { x: 3 * COL, y: 1.5 * ROW } });
  const m2 = graph.addNode("mul", { label: "w2·h", position: { x: 4 * COL, y: 2 * ROW } });
  const pre2 = graph.addNode("add", { label: "z2", position: { x: 5 * COL, y: 2.5 * ROW } });
  const out = graph.addNode("tanh", { label: "ŷ", position: { x: 6 * COL, y: 3 * ROW } });

  graph.connect(x.id, m1.id);
  graph.connect(w1.id, m1.id);
  graph.connect(m1.id, pre1.id);
  graph.connect(b1.id, pre1.id);
  graph.connect(pre1.id, hidden.id);
  graph.connect(hidden.id, m2.id);
  graph.connect(w2.id, m2.id);
  graph.connect(m2.id, pre2.id);
  graph.connect(b2.id, pre2.id);
  graph.connect(pre2.id, out.id);
  graph.setOutput(out.id);
  return graph;
}

/**
 * 2→3→1 perceptron unrolled into scalar ops: hⱼ = tanh(wⱼ1·x1 + wⱼ2·x2 + bⱼ)
 * and ŷ = Σⱼ vⱼ·hⱼ + c. Both inputs fan out to three multiplications each.
 */
function buildMlp(): AutogradGraph {
  const weights = [
    [0.5, -1],
    [-0.5, 1],
    [1, 0.5],
  ];
  const biases = [0.25, -0.25, 0];
  const readout = [1, -1, 0.5];

  resetAutogradIdCounter(1);
  const graph = new AutogradGraph();
  const at = (col: number, row: number): AutogradPosition => ({ x: col * COL, y: row * ROW });

  const x1 = graph.addNode("input", { value: 1, label: "x1", position: at(1, 0.5) });
  const x2 = graph.addNode("input", { value: 0.5, label: "x2", position: at(1, 6.5) });
  const x = [x1, x2];
  const c = graph.addNode("input", { value: 0.1, label: "c", position: at(6, 4.5) });
  const y = graph.addNode("add", { label: "ŷ", position: at(7, 3.5) });

  // Hidden neuron j owns rows 2j+1 (x1 branch) and 2j+2 (x2 branch).
  for (let j = 0; j < weights.length; j++) {
    const n = j + 1;
    const top = 1 + 2 * j;
    const sum = graph.addNode("add", { label: `z${n}`, position: at(4, top + 0.5) });
    const hidden = graph.addNode("tanh", { label: `h${n}`, position: at(5, top + 0.5) });
    const v = graph.addNode("input", { value: readout[j], label: `v${n}`, position: at(5, top + 1.5) });
    const prod = graph.addNode("mul", { label: `v${n}·h${n}`, position: at(6, top + 0.5) });
    const bias = graph.addNode("input", { value: biases[j], label: `b${n}`, position: at(3, top + 1.5) });

    for (let i = 0; i < x.length; i++) {
      const w = graph.addNode("input", { value: weights[j][i], label: `w${n}${i + 1}`, position: at(0, top + i) });
      const m = graph.addNode("mul", { label: `w${n}${i + 1}·x${i + 1}`, position: at(2, top + i) });
      graph.connect(w.id, m.id);
      graph.connect(x[i].id, m.id);
      graph.connect(m.id, sum.id);
    }
    graph.connect(bias.id, sum.id);
    graph.connect(sum.id, hidden.id);
    graph.connect(hidden.id, prod.id);
    graph.connect(v.id, prod.id);
    graph.connect(prod.id, y.id);
  }
  graph.connect(c.id, y.id);
  graph.setOutput(y.id);
  return graph;
}

/**
 * out = Σᵢ softmax(q·kᵢ)ᵢ · vᵢ over two positions — scalar attention. The query
 * feeds both scores and the normalizer Σeᵢ feeds both attention weights.
 */
function buildAttention(): AutogradGraph {
  resetAutogradIdCounter(1);
  const graph = new AutogradGraph();
  const k1 = graph.addNode("input", { value: 1, label: "k1", position: { x: 0, y: 0 } });
  const k2 = graph.addNode("input", { value: -1, label: "k2", position: { x: 0, y: 4 * ROW } });
  const q = graph.addNode("input", { value: 1, label: "q", position: { x: 0, y: 2 * ROW } });
  const v1 = graph.addNode("input", { value: 2, label: "v1", position: { x: 4 * COL, y: 1.5 * ROW } });
  const v2 = graph.addNode("input", { value: -1, label: "v2", position: { x: 4 * COL, y: 2.5 * ROW } });

  const s1 = graph.addNode("mul", { label: "q·k1", position: { x: COL, y: ROW / 2 } });
  const s2 = graph.addNode("mul", { label: "q·k2", position: { x: COL, y: 3.5 * ROW } });
  const e1 = graph.addNode("exp", { label: "e^s1", position: { x: 2 * COL, y: ROW / 2 } });
  const e2 = graph.addNode("exp", { label: "e^s2", position: { x: 2 * COL, y: 3.5 * ROW } });
  const total = graph.addNode("add", { label: "Σe", position: { x: 3 * COL, y: 2 * ROW } });
  const a1 = graph.addNode("div", { label: "α1", position: { x: 4 * COL, y: ROW / 2 } });
  const a2 = graph.addNode("div", { label: "α2", position: { x: 4 * COL, y: 3.5 * ROW } });
  const m1 = graph.addNode("mul", { label: "α1·v1", position: { x: 5 * COL, y: ROW / 2 } });
  const m2 = graph.addNode("mul", { label: "α2·v2", position: { x: 5 * COL, y: 3.5 * ROW } });
  const out = graph.addNode("add", { label: "out", position: { x: 6 * COL, y: 2 * ROW } });

  graph.connect(q.id, s1.id);
  graph.connect(k1.id, s1.id);
  graph.connect(q.id, s2.id);
  graph.connect(k2.id, s2.id);
  graph.connect(s1.id, e1.id);
  graph.connect(s2.id, e2.id);
  graph.connect(e1.id, total.id);
  graph.connect(e2.id, total.id);
  // div is positional: the numerator goes in first.
  graph.connect(e1.id, a1.id);
  graph.connect(total.id, a1.id);
  graph.connect(e2.id, a2.id);
  graph.connect(total.id, a2.id);
  graph.connect(a1.id, m1.id);
  graph.connect(v1.id, m1.id);
  graph.connect(a2.id, m2.id);
  graph.connect(v2.id, m2.id);
  graph.connect(m1.id, out.id);
  graph.connect(m2.id, out.id);
  graph.setOutput(out.id);
  return graph;
}

const PRESET_BUILDERS: Record<AutogradPresetId, () => AutogradGraph> = {
  expr: buildExpr,
  neuron: buildNeuron,
  sigmoid: buildSigmoid,
  hinge: buildHinge,
  twolayer: buildTwoLayer,
  mlp: buildMlp,
  attention: buildAttention,
};

/**
 * Presets define topology and leaf values only: nothing is computed until the
 * user runs a forward pass.
 */
export function buildAutogradPreset(preset: AutogradPresetId): AutogradGraph {
  return PRESET_BUILDERS[preset]();
}

export const AUTOGRAD_PRESETS: AutogradPresetId[] = [
  "expr",
  "neuron",
  "sigmoid",
  "hinge",
  "twolayer",
  "mlp",
  "attention",
];
