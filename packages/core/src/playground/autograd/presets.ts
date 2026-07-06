import { AutogradGraph, resetAutogradIdCounter } from "./graph";

export type AutogradPresetId = "expr" | "neuron";

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
  graph.forward();
  graph.backward();
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
  graph.forward();
  graph.backward();
  return graph;
}

export function buildAutogradPreset(preset: AutogradPresetId): AutogradGraph {
  return preset === "neuron" ? buildNeuron() : buildExpr();
}

export const AUTOGRAD_PRESETS: AutogradPresetId[] = ["expr", "neuron"];
