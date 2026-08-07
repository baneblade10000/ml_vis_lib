import type { ActivationFunction, ErrorFunction, Link, RegularizationFunction as RegFn } from "../nn";
import {
  Activations,
  createGraphLink,
  GraphNode,
  resetGraphDerivatives,
  RegularizationFunction,
} from "./runtime";
import type { GraphEdgeDef, GraphNodeDef, GraphPosition, GraphSnapshot } from "./types";
import { optimizerDelta, type PlaygroundOptimizerId } from "../../optimizers";

let nextNodeId = 1;

export function resetGraphIdCounter(start = 1): void {
  nextNodeId = start;
}

/** Advance the auto-id counter past all numeric ids already present in the graph. */
export function syncGraphIdCounterFromGraph(graph: ComputationalGraph): void {
  let maxNumeric = 0;
  for (const id of graph.nodes.keys()) {
    const num = Number.parseInt(id, 10);
    if (!Number.isNaN(num) && num > maxNumeric) maxNumeric = num;
  }
  resetGraphIdCounter(maxNumeric + 1);
}

function allocId(): string {
  return String(nextNodeId++);
}

export class ComputationalGraph {
  nodes = new Map<string, GraphNode>();
  inputIds: string[] = [];
  outputId = "";
  positions = new Map<string, GraphPosition>();
  private topoOrder: string[] = [];
  regularization: RegFn | null = null;

  static fromSnapshot(snapshot: GraphSnapshot, regularization: RegFn | null = null): ComputationalGraph {
    const graph = new ComputationalGraph();
    graph.regularization = regularization;
    resetGraphIdCounter(1000);

    for (const def of snapshot.nodes) {
      const node = new GraphNode(def.id, def.kind, def.activation, false, def.label);
      node.bias = def.bias;
      graph.nodes.set(def.id, node);
    }

    graph.inputIds = [...snapshot.inputIds];
    graph.outputId = snapshot.outputId;

    for (const [id, pos] of Object.entries(snapshot.positions)) {
      graph.positions.set(id, { ...pos });
    }

    for (const edge of snapshot.edges) {
      const source = graph.nodes.get(edge.source);
      const dest = graph.nodes.get(edge.target);
      if (!source || !dest) continue;
      const link = createGraphLink(source, dest, regularization);
      link.weight = edge.weight;
      link.id = edge.id;
    }

    graph.recomputeTopoOrder();
    return graph;
  }

  toSnapshot(): GraphSnapshot {
    const nodes: GraphNodeDef[] = [];
    for (const node of this.nodes.values()) {
      nodes.push({
        id: node.id,
        kind: node.kind,
        activation: node.activation,
        bias: node.bias,
        label: node.label,
      });
    }

    const edges: GraphEdgeDef[] = [];
    const seen = new Set<string>();
    for (const node of this.nodes.values()) {
      for (const link of node.outputs) {
        if (seen.has(link.id)) continue;
        seen.add(link.id);
        edges.push({
          id: link.id,
          source: link.source.id,
          target: link.dest.id,
          weight: link.weight,
        });
      }
    }

    const positions: Record<string, GraphPosition> = {};
    for (const [id, pos] of this.positions) {
      positions[id] = { ...pos };
    }

    return {
      nodes,
      edges,
      inputIds: [...this.inputIds],
      outputId: this.outputId,
      positions,
    };
  }

  getNode(id: string): GraphNode | undefined {
    return this.nodes.get(id);
  }

  getOutputNode(): GraphNode {
    const node = this.nodes.get(this.outputId);
    if (!node) throw new Error("Output node not found");
    return node;
  }

  getTopoOrder(): string[] {
    return this.topoOrder;
  }

  getAllLinks(): Link[] {
    const links: Link[] = [];
    const seen = new Set<string>();
    for (const node of this.nodes.values()) {
      for (const link of node.outputs) {
        if (!seen.has(link.id)) {
          seen.add(link.id);
          links.push(link);
        }
      }
    }
    return links;
  }

  setPosition(nodeId: string, pos: GraphPosition): void {
    this.positions.set(nodeId, { ...pos });
  }

  addInputNode(id: string, label?: string, position?: GraphPosition): GraphNode {
    const node = new GraphNode(id, "input", Activations.LINEAR, true, label);
    this.nodes.set(id, node);
    if (!this.inputIds.includes(id)) {
      this.inputIds.push(id);
    }
    if (position) this.positions.set(id, position);
    this.recomputeTopoOrder();
    return node;
  }

  addNode(
    kind: GraphNode["kind"],
    activation: ActivationFunction,
    options?: { id?: string; label?: string; position?: GraphPosition; initZero?: boolean },
  ): GraphNode {
    const id = options?.id ?? allocId();
    const node = new GraphNode(id, kind, activation, options?.initZero, options?.label);
    this.nodes.set(id, node);
    if (kind === "output") {
      this.outputId = id;
    }
    if (options?.position) this.positions.set(id, options.position);
    this.recomputeTopoOrder();
    return node;
  }

  connect(sourceId: string, targetId: string, weight?: number): Link | null {
    const source = this.nodes.get(sourceId);
    const dest = this.nodes.get(targetId);
    if (!source || !dest) return null;
    if (sourceId === targetId) return null;
    if (source.kind === "output" || dest.kind === "input") return null;
    if (this.wouldCreateCycle(sourceId, targetId)) return null;

    const existing = source.outputs.find((l) => l.dest.id === targetId);
    if (existing) return existing;

    const link = createGraphLink(source, dest, this.regularization, false, weight);
    this.recomputeTopoOrder();
    return link;
  }

  removeNode(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (!node) return;
    if (node.kind === "input") {
      this.inputIds = this.inputIds.filter((id) => id !== nodeId);
    }
    if (this.outputId === nodeId) {
      this.outputId = "";
    }

    for (const link of [...node.inputLinks]) {
      this.removeLink(link.id);
    }
    for (const link of [...node.outputs]) {
      this.removeLink(link.id);
    }
    this.nodes.delete(nodeId);
    this.positions.delete(nodeId);
    this.recomputeTopoOrder();
  }

  removeLink(linkId: string): void {
    for (const node of this.nodes.values()) {
      node.inputLinks = node.inputLinks.filter((l) => l.id !== linkId);
      node.outputs = node.outputs.filter((l) => l.id !== linkId);
    }
    this.recomputeTopoOrder();
  }

  validate(): { valid: boolean; error?: string } {
    if (!this.outputId || !this.nodes.has(this.outputId)) {
      return { valid: false, error: "Missing output node" };
    }
    if (this.inputIds.length === 0) {
      return { valid: false, error: "No input nodes" };
    }
    try {
      this.recomputeTopoOrder();
    } catch (e) {
      return { valid: false, error: e instanceof Error ? e.message : "Invalid graph" };
    }
    return { valid: true };
  }

  recomputeTopoOrder(): void {
    const order: string[] = [];
    const visited = new Set<string>();
    const visiting = new Set<string>();

    const visit = (id: string): void => {
      if (visited.has(id)) return;
      if (visiting.has(id)) {
        throw new Error("Cycle detected in network graph");
      }
      visiting.add(id);
      const node = this.nodes.get(id);
      if (node) {
        for (const link of node.outputs) {
          visit(link.dest.id);
        }
      }
      visiting.delete(id);
      visited.add(id);
      order.push(id);
    };

    for (const id of this.inputIds) {
      visit(id);
    }
    for (const id of this.nodes.keys()) {
      visit(id);
    }

    this.topoOrder = order.reverse();
  }

  private wouldCreateCycle(sourceId: string, targetId: string): boolean {
    const visited = new Set<string>();
    const stack = [targetId];
    while (stack.length) {
      const id = stack.pop()!;
      if (id === sourceId) return true;
      if (visited.has(id)) continue;
      visited.add(id);
      const node = this.nodes.get(id);
      if (!node) continue;
      for (const link of node.outputs) {
        stack.push(link.dest.id);
      }
    }
    return false;
  }

  /** Convert to layered Node[][] for backward-compatible consumers. */
  toLayeredNetwork(): GraphNode[][] {
    const layers: GraphNode[][] = [];
    const assigned = new Set<string>();
    const nodeLayer = new Map<string, number>();

    for (const id of this.inputIds) {
      nodeLayer.set(id, 0);
      assigned.add(id);
    }

    let changed = true;
    while (changed) {
      changed = false;
      for (const id of this.topoOrder) {
        if (assigned.has(id)) continue;
        const node = this.nodes.get(id);
        if (!node) continue;
        const preds = node.inputLinks.map((l) => l.source.id);
        if (preds.length === 0 || preds.every((p) => assigned.has(p))) {
          const maxPred = preds.length ? Math.max(...preds.map((p) => nodeLayer.get(p) ?? 0)) : 0;
          nodeLayer.set(id, maxPred + 1);
          assigned.add(id);
          changed = true;
        }
      }
    }

    const maxLayer = Math.max(0, ...nodeLayer.values());
    for (let i = 0; i <= maxLayer; i++) {
      layers.push([]);
    }

    for (const id of this.topoOrder) {
      const layer = nodeLayer.get(id) ?? 0;
      const node = this.nodes.get(id);
      if (node) layers[layer].push(node);
    }

    return layers;
  }
}

export function forwardPropGraph(
  graph: ComputationalGraph,
  inputs: number[],
  resetDerivatives = true,
): number {
  if (inputs.length !== graph.inputIds.length) {
    throw new Error("Input count mismatch");
  }

  if (resetDerivatives) {
    resetGraphDerivatives(graph.nodes.values());
  }

  graph.inputIds.forEach((id, i) => {
    const node = graph.nodes.get(id);
    if (node) node.output = inputs[i];
  });

  for (const id of graph.getTopoOrder()) {
    const node = graph.nodes.get(id);
    if (!node || node.kind === "input") continue;
    node.updateOutput();
  }

  return graph.getOutputNode().output;
}

export function backPropGraph(graph: ComputationalGraph, target: number, errorFunc: ErrorFunction): void {
  const outputNode = graph.getOutputNode();
  outputNode.outputDer = errorFunc.der(outputNode.output, target);

  const order = graph.getTopoOrder();
  for (let i = order.length - 1; i >= 0; i--) {
    const node = graph.nodes.get(order[i]);
    if (!node || node.kind === "input") continue;

    if (node !== outputNode) {
      node.outputDer = 0;
      for (const link of node.outputs) {
        node.outputDer += link.weight * link.dest.inputDer;
      }
    }

    node.inputDer = node.outputDer * node.activation.der(node.totalInput);
    node.accInputDer += node.inputDer;
    node.numAccumulatedDers++;

    for (const link of node.inputLinks) {
      if (link.isDead) continue;
      link.errorDer = node.inputDer * link.source.output;
      link.accErrorDer += link.errorDer;
      link.numAccumulatedDers++;
    }
  }
}

export function updateWeightsGraph(
  graph: ComputationalGraph,
  learningRate: number,
  regularizationRate: number,
  optimizer: PlaygroundOptimizerId = "SGD",
  optStep = 1,
): void {
  for (const id of graph.getTopoOrder()) {
    const node = graph.nodes.get(id);
    if (!node || node.kind === "input") continue;

    if (node.numAccumulatedDers > 0) {
      const gBias = node.accInputDer / node.numAccumulatedDers;
      node.bias -= optimizerDelta(gBias, node, optimizer, learningRate, optStep);
      node.accInputDer = 0;
      node.numAccumulatedDers = 0;
    }

    for (const link of node.inputLinks) {
      if (link.isDead) continue;
      const regulDer = link.regularization ? link.regularization.der(link.weight) : 0;
      if (link.numAccumulatedDers > 0) {
        link.lastGradient = link.accErrorDer / link.numAccumulatedDers;
        const gData = link.accErrorDer / link.numAccumulatedDers;
        // Regularization is folded into the effective gradient (same as CNN path).
        const gEff = gData + regularizationRate * regulDer;
        const prev = link.weight;
        const next = prev - optimizerDelta(gEff, link, optimizer, learningRate, optStep);
        if (link.regularization === RegularizationFunction.L1 && prev * next < 0) {
          link.weight = 0;
          link.isDead = true;
        } else {
          link.weight = next;
        }
        link.accErrorDer = 0;
        link.numAccumulatedDers = 0;
      }
    }
  }
}

export function forEachGraphNode(
  graph: ComputationalGraph,
  ignoreInputs: boolean,
  accessor: (node: GraphNode) => void,
): void {
  for (const id of graph.getTopoOrder()) {
    const node = graph.nodes.get(id);
    if (!node) continue;
    if (ignoreInputs && node.kind === "input") continue;
    accessor(node);
  }
}
