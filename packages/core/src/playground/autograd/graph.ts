import { evaluateOp, localDerivatives } from "./ops";
import { OP_SPECS, type AutogradOp, type AutogradPosition, type AutogradSnapshot } from "./types";

export class AutogradNode {
  id: string;
  op: AutogradOp;
  label?: string;
  /** Editable scalar for leaf nodes; computed result for operations after forward(). */
  value: number;
  /** d(output)/d(this) after backward(). */
  grad = 0;

  constructor(id: string, op: AutogradOp, value = 0, label?: string) {
    this.id = id;
    this.op = op;
    this.value = value;
    this.label = label;
  }

  get isLeaf(): boolean {
    return OP_SPECS[this.op].leaf;
  }
}

export class AutogradEdge {
  id: string;
  source: string;
  target: string;
  /** Local partial d(target.value)/d(source.value) after backward(). */
  localDer = 0;

  constructor(id: string, source: string, target: string) {
    this.id = id;
    this.source = source;
    this.target = target;
  }
}

let nextAutogradId = 1;

export function resetAutogradIdCounter(start = 1): void {
  nextAutogradId = start;
}

function allocAutogradId(): string {
  return String(nextAutogradId++);
}

export class AutogradGraph {
  nodes = new Map<string, AutogradNode>();
  edges = new Map<string, AutogradEdge>();
  outputId = "";
  positions = new Map<string, AutogradPosition>();
  private topoOrder: string[] = [];

  getNode(id: string): AutogradNode | undefined {
    return this.nodes.get(id);
  }

  getTopoOrder(): string[] {
    return this.topoOrder;
  }

  /** Incoming edges of a node, kept in insertion order (input slot order). */
  inputEdges(nodeId: string): AutogradEdge[] {
    return [...this.edges.values()].filter((e) => e.target === nodeId);
  }

  outputEdges(nodeId: string): AutogradEdge[] {
    return [...this.edges.values()].filter((e) => e.source === nodeId);
  }

  addNode(op: AutogradOp, options?: { id?: string; value?: number; label?: string; position?: AutogradPosition }): AutogradNode {
    const id = options?.id ?? allocAutogradId();
    const node = new AutogradNode(id, op, options?.value ?? 0, options?.label);
    this.nodes.set(id, node);
    if (options?.position) this.positions.set(id, { ...options.position });
    this.recomputeTopoOrder();
    return node;
  }

  setPosition(nodeId: string, pos: AutogradPosition): void {
    this.positions.set(nodeId, { ...pos });
  }

  /**
   * Attempt to connect source -> target. Rejects self-loops, cycles, duplicate
   * edges, connections into leaf nodes, and edges exceeding the target arity.
   */
  connect(sourceId: string, targetId: string): AutogradEdge | null {
    if (sourceId === targetId) return null;
    const source = this.nodes.get(sourceId);
    const target = this.nodes.get(targetId);
    if (!source || !target) return null;
    if (target.isLeaf) return null;

    const spec = OP_SPECS[target.op];
    const existingInputs = this.inputEdges(targetId);
    if (existingInputs.length >= spec.arity) return null;
    if (existingInputs.some((e) => e.source === sourceId)) return null;
    if (this.wouldCreateCycle(sourceId, targetId)) return null;

    const edge = new AutogradEdge(allocAutogradId(), sourceId, targetId);
    this.edges.set(edge.id, edge);
    this.recomputeTopoOrder();
    return edge;
  }

  removeEdge(edgeId: string): void {
    this.edges.delete(edgeId);
    this.recomputeTopoOrder();
  }

  removeNode(nodeId: string): void {
    for (const edge of [...this.edges.values()]) {
      if (edge.source === nodeId || edge.target === nodeId) {
        this.edges.delete(edge.id);
      }
    }
    this.nodes.delete(nodeId);
    this.positions.delete(nodeId);
    if (this.outputId === nodeId) {
      this.outputId = this.pickOutputId();
    }
    this.recomputeTopoOrder();
  }

  /** A sink node (no outgoing edges) makes the most natural output. */
  private pickOutputId(): string {
    for (const id of this.topoOrder.length ? this.topoOrder : [...this.nodes.keys()]) {
      const node = this.nodes.get(id);
      if (node && !node.isLeaf && this.outputEdges(id).length === 0) return id;
    }
    const last = [...this.nodes.keys()].at(-1);
    return last ?? "";
  }

  setOutput(nodeId: string): void {
    if (this.nodes.has(nodeId)) this.outputId = nodeId;
  }

  private wouldCreateCycle(sourceId: string, targetId: string): boolean {
    // Adding source->target creates a cycle iff source is reachable from target.
    const visited = new Set<string>();
    const stack = [targetId];
    while (stack.length) {
      const id = stack.pop()!;
      if (id === sourceId) return true;
      if (visited.has(id)) continue;
      visited.add(id);
      for (const edge of this.outputEdges(id)) {
        stack.push(edge.target);
      }
    }
    return false;
  }

  recomputeTopoOrder(): void {
    const order: string[] = [];
    const visited = new Set<string>();
    const visiting = new Set<string>();

    const visit = (id: string): void => {
      if (visited.has(id)) return;
      if (visiting.has(id)) throw new Error("Cycle detected in computational graph");
      visiting.add(id);
      for (const edge of this.outputEdges(id)) {
        visit(edge.target);
      }
      visiting.delete(id);
      visited.add(id);
      order.push(id);
    };

    for (const id of this.nodes.keys()) visit(id);
    this.topoOrder = order.reverse();
    if (!this.outputId || !this.nodes.has(this.outputId)) {
      this.outputId = this.pickOutputId();
    }
  }

  /** Forward pass: compute `value` for every operation node in topological order. */
  forward(): void {
    for (const id of this.topoOrder) {
      const node = this.nodes.get(id);
      if (!node || node.isLeaf) continue;
      const inputs = this.inputEdges(id).map((e) => this.nodes.get(e.source)?.value ?? 0);
      node.value = evaluateOp(node.op, inputs);
    }
  }

  /**
   * Reverse-mode autodiff: seed the output gradient to 1 and propagate through
   * the reversed topological order, accumulating node grads and per-edge local
   * derivatives (chain rule).
   */
  backward(): void {
    for (const node of this.nodes.values()) node.grad = 0;
    for (const edge of this.edges.values()) edge.localDer = 0;
    if (!this.outputId) return;

    const output = this.nodes.get(this.outputId);
    if (output) output.grad = 1;

    for (let i = this.topoOrder.length - 1; i >= 0; i--) {
      const node = this.nodes.get(this.topoOrder[i]);
      if (!node || node.isLeaf) continue;
      const inEdges = this.inputEdges(node.id);
      const inputs = inEdges.map((e) => this.nodes.get(e.source)?.value ?? 0);
      const ders = localDerivatives(node.op, inputs, node.value);
      inEdges.forEach((edge, idx) => {
        const local = ders[idx] ?? 0;
        edge.localDer = local;
        const source = this.nodes.get(edge.source);
        if (source) source.grad += node.grad * local;
      });
    }
  }

  toSnapshot(): AutogradSnapshot {
    return {
      nodes: [...this.nodes.values()].map((n) => ({ id: n.id, op: n.op, label: n.label, value: n.value })),
      edges: [...this.edges.values()].map((e) => ({ id: e.id, source: e.source, target: e.target })),
      outputId: this.outputId,
      positions: Object.fromEntries([...this.positions.entries()].map(([id, p]) => [id, { ...p }])),
    };
  }

  static fromSnapshot(snapshot: AutogradSnapshot): AutogradGraph {
    const graph = new AutogradGraph();
    let maxId = 0;
    for (const def of snapshot.nodes) {
      graph.addNode(def.op, { id: def.id, value: def.value, label: def.label, position: snapshot.positions[def.id] });
      const num = Number.parseInt(def.id, 10);
      if (!Number.isNaN(num)) maxId = Math.max(maxId, num);
    }
    for (const def of snapshot.edges) {
      const edge = new AutogradEdge(def.id, def.source, def.target);
      graph.edges.set(def.id, edge);
      const num = Number.parseInt(def.id, 10);
      if (!Number.isNaN(num)) maxId = Math.max(maxId, num);
    }
    graph.outputId = snapshot.outputId;
    resetAutogradIdCounter(maxId + 1);
    graph.recomputeTopoOrder();
    return graph;
  }
}
