import { evaluateOp, localDerivatives } from "./ops";
import {
  OP_SPECS,
  type AutogradOp,
  type AutogradPosition,
  type AutogradSnapshot,
  type AutogradStep,
} from "./types";

export class AutogradNode {
  id: string;
  op: AutogradOp;
  label?: string;
  /**
   * Editable scalar for leaf nodes; the result of the operation once the
   * forward pass reached it — see {@link AutogradGraph.isComputed}.
   */
  value: number;
  /**
   * d(output)/d(this) as far as the backward pass has accumulated it — see
   * {@link AutogradGraph.hasGrad}.
   */
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
  /**
   * Which of the target's input ports the edge feeds, counting from 0 — for
   * a positional operation (sub, div), which of its two operands.
   */
  slot: number;
  /**
   * Local partial d(target.value)/d(source.value) once the backward pass went
   * through the target — see {@link AutogradGraph.hasLocalDer}.
   */
  localDer = 0;

  constructor(id: string, source: string, target: string, slot = 0) {
    this.id = id;
    this.source = source;
    this.target = target;
    this.slot = slot;
  }
}

let nextAutogradId = 1;

export function resetAutogradIdCounter(start = 1): void {
  nextAutogradId = start;
}

function allocAutogradId(): string {
  return String(nextAutogradId++);
}

/**
 * An n-ary operation (add, mul) shows this many input ports before anything
 * is connected to it, like a binary one, and one more than its highest fed
 * port afterwards.
 */
const NARY_MIN_PORTS = 2;

export class AutogradGraph {
  nodes = new Map<string, AutogradNode>();
  edges = new Map<string, AutogradEdge>();
  outputId = "";
  positions = new Map<string, AutogradPosition>();
  private topoOrder: string[] = [];
  /** Operation nodes whose value is current. */
  private computedIds = new Set<string>();
  /** Operation nodes the backward pass has pushed a gradient through. */
  private backwardIds = new Set<string>();
  /** Whether the backward pass has seeded the output gradient. */
  private backwardSeeded = false;
  /** Nodes whose gradient the backward pass has seeded or accumulated so far. */
  private gradIds = new Set<string>();
  private last: AutogradStep | null = null;

  getNode(id: string): AutogradNode | undefined {
    return this.nodes.get(id);
  }

  getTopoOrder(): string[] {
    return this.topoOrder;
  }

  /** Incoming edges of a node through its input ports, lowest port first. */
  inputEdges(nodeId: string): AutogradEdge[] {
    return [...this.edges.values()].filter((e) => e.target === nodeId).sort((a, b) => a.slot - b.slot);
  }

  /**
   * How many input ports a node has: a leaf none, an operation as many as its
   * arity, and an n-ary operation (add, mul) at least two and one more than
   * its highest fed port.
   */
  inputPortCount(nodeId: string): number {
    const node = this.nodes.get(nodeId);
    if (!node || node.isLeaf) return 0;
    const arity = OP_SPECS[node.op].arity;
    if (arity !== Infinity) return arity;
    let highest = -1;
    for (const edge of this.edges.values()) {
      if (edge.target === nodeId) highest = Math.max(highest, edge.slot);
    }
    return Math.max(NARY_MIN_PORTS, highest + 1);
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
   * Attempt to connect source -> target through one of the target's input
   * ports, counting from 0. The given port takes the edge, or without `slot`
   * the first free one. An operation of fixed arity (neg, sub, div, ...) whose
   * port is fed already hands it to the new source. An n-ary operation (add,
   * mul) never loses an operand: a wire dropped on a fed port goes to the
   * first free one, or to a new port. Rejects self-loops, cycles, connections
   * into leaf nodes, ports the operation does not have, a fixed-arity
   * operation that is fully fed when no port is given, and a source that
   * already feeds the port it would take over.
   */
  connect(sourceId: string, targetId: string, slot?: number): AutogradEdge | null {
    if (sourceId === targetId) return null;
    const source = this.nodes.get(sourceId);
    const target = this.nodes.get(targetId);
    if (!source || !target || target.isLeaf) return null;
    if (this.wouldCreateCycle(sourceId, targetId)) return null;

    const ports = this.inputPortCount(targetId);
    const fed = new Set(this.inputEdges(targetId).map((edge) => edge.slot));
    const isPort = (port: number): boolean => Number.isInteger(port) && port >= 0 && port < ports;
    let firstFree = 0;
    while (firstFree < ports && fed.has(firstFree)) firstFree += 1;

    let port: number;
    if (OP_SPECS[target.op].arity === Infinity) {
      // A free port the wire names, else the first free one, else a new one.
      port = firstFree < ports ? firstFree : ports;
      if (slot !== undefined && isPort(slot) && !fed.has(slot)) port = slot;
    } else {
      // Full with no port named, or a port the operation does not have.
      port = slot ?? firstFree;
      if (!isPort(port)) return null;
    }

    const occupant = this.inputEdges(targetId).find((edge) => edge.slot === port);
    if (occupant?.source === sourceId) return null;
    if (occupant) this.edges.delete(occupant.id);

    const edge = new AutogradEdge(allocAutogradId(), sourceId, targetId, port);
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

  /** Re-derive the topological order after a topology change; every value and gradient computed so far goes stale with it. */
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
    this.resetPasses();
  }

  /**
   * Forget all progress of both passes, leaving the graph as before either
   * ran: the values and gradients computed so far stay in place, but are no
   * longer current. Topology changes do this on their own; changing a leaf
   * value has to call it explicitly.
   */
  resetPasses(): void {
    this.computedIds.clear();
    this.backwardIds.clear();
    this.backwardSeeded = false;
    this.gradIds.clear();
    this.last = null;
  }

  /** Whether the node's value is current: a leaf's always is, an operation's once the forward pass reached it. */
  isComputed(nodeId: string): boolean {
    const node = this.nodes.get(nodeId);
    return !!node && (node.isLeaf || this.computedIds.has(nodeId));
  }

  /**
   * Whether the node's gradient is known: once the backward pass seeded or
   * accumulated it — for every node once the pass is complete.
   */
  hasGrad(nodeId: string): boolean {
    return this.nodes.has(nodeId) && (this.gradIds.has(nodeId) || this.isBackwardComplete());
  }

  /** Whether the edge's local derivative is known: once the backward pass went through the operation it feeds. */
  hasLocalDer(edgeId: string): boolean {
    const edge = this.edges.get(edgeId);
    return !!edge && this.backwardIds.has(edge.target);
  }

  /**
   * Operation nodes a forward step takes next: every one without a value
   * whose inputs all have theirs, in topological order. None of them depends
   * on another, so the step computes all of them at once.
   */
  private nextForwardIds(): string[] {
    const ready: string[] = [];
    for (const id of this.topoOrder) {
      const node = this.nodes.get(id);
      if (!node || node.isLeaf || this.computedIds.has(id)) continue;
      if (this.inputEdges(id).every((edge) => this.isComputed(edge.source))) ready.push(id);
    }
    return ready;
  }

  /**
   * Operation nodes a backward step takes next: every one that is not done
   * yet and whose gradient is final because all of its consumers are, in
   * reverse topological order. As with a forward step, none depends on another.
   */
  private nextBackwardIds(): string[] {
    const ready: string[] = [];
    for (let i = this.topoOrder.length - 1; i >= 0; i--) {
      const id = this.topoOrder[i];
      const node = this.nodes.get(id);
      if (!node || node.isLeaf || this.backwardIds.has(id)) continue;
      if (this.outputEdges(id).every((edge) => this.backwardIds.has(edge.target))) ready.push(id);
    }
    return ready;
  }

  private isBackwardComplete(): boolean {
    return this.backwardSeeded && this.nextBackwardIds().length === 0;
  }

  /**
   * What the next {@link step} does: every operation node that is ready, in
   * the forward pass until it is complete, then in the backward pass; null
   * once both are.
   */
  get nextStep(): AutogradStep | null {
    const forwardIds = this.nextForwardIds();
    if (forwardIds.length > 0) return { ids: forwardIds, pass: "forward" };
    const backwardIds = this.nextBackwardIds();
    return backwardIds.length === 0 ? null : { ids: backwardIds, pass: "backward" };
  }

  /** The operation nodes the last {@link step} worked on; null after a whole pass or any change to the graph. */
  get lastStep(): AutogradStep | null {
    return this.last;
  }

  private countOperations(): number {
    let total = 0;
    for (const node of this.nodes.values()) {
      if (!node.isLeaf) total += 1;
    }
    return total;
  }

  /** How far the forward pass has got: operation nodes with a value so far, and in total. */
  forwardProgress(): { done: number; total: number } {
    return { done: this.computedIds.size, total: this.countOperations() };
  }

  /** How far the backward pass has got: operation nodes it pushed a gradient through so far, and in total. */
  backwardProgress(): { done: number; total: number } {
    return { done: this.backwardIds.size, total: this.countOperations() };
  }

  /**
   * The values reaching a node through its ports, each at the index of its
   * port. An unfed port leaves a hole, which an operation reads as a missing
   * operand.
   */
  private inputValues(nodeId: string): number[] {
    const values: number[] = [];
    for (const edge of this.inputEdges(nodeId)) {
      values[edge.slot] = this.nodes.get(edge.source)?.value ?? 0;
    }
    return values;
  }

  /** Write `value` of one operation node from its current inputs. */
  private evaluate(node: AutogradNode): void {
    node.value = evaluateOp(node.op, this.inputValues(node.id));
  }

  /**
   * Push `node`'s gradient through its operation: set the local derivative of
   * each input edge and add to the inputs' gradients. A source that feeds
   * several ports collects the gradient of each.
   */
  private propagate(node: AutogradNode): void {
    const inEdges = this.inputEdges(node.id);
    const ders = localDerivatives(node.op, this.inputValues(node.id), node.value);
    for (const edge of inEdges) {
      const local = ders[edge.slot] ?? 0;
      edge.localDer = local;
      const source = this.nodes.get(edge.source);
      if (source) {
        source.grad += node.grad * local;
        this.gradIds.add(source.id);
      }
    }
  }

  /** Start the backward pass from scratch: zero every gradient, then seed the output's with d(output)/d(output) = 1. */
  private startBackward(): void {
    for (const node of this.nodes.values()) node.grad = 0;
    for (const edge of this.edges.values()) edge.localDer = 0;
    this.gradIds.clear();
    const output = this.outputId ? this.nodes.get(this.outputId) : undefined;
    if (output) {
      output.grad = 1;
      this.gradIds.add(output.id);
    }
    this.backwardSeeded = true;
  }

  /** Forward pass: compute `value` for every operation node in topological order. */
  forward(): void {
    this.resetPasses();
    for (const id of this.topoOrder) {
      const node = this.nodes.get(id);
      if (!node || node.isLeaf) continue;
      this.evaluate(node);
      this.computedIds.add(id);
    }
  }

  /**
   * Reverse-mode autodiff over the values of the last forward pass: seed the
   * output gradient to 1 and propagate through the reversed topological
   * order, accumulating node grads and per-edge local derivatives (chain rule).
   */
  backward(): void {
    this.backwardIds.clear();
    this.last = null;
    this.startBackward();
    for (let i = this.topoOrder.length - 1; i >= 0; i--) {
      const node = this.nodes.get(this.topoOrder[i]);
      if (!node || node.isLeaf) continue;
      this.propagate(node);
      this.backwardIds.add(node.id);
    }
  }

  /**
   * One step of whichever pass is up next, through every operation node that
   * is ready, so independent operations advance together. A forward step
   * computes each node whose inputs all have a value. A backward step seeds
   * the output gradient if it is the first, then pushes the gradient through
   * each node whose consumers are all done. Returns the step, or null once
   * both passes are complete.
   */
  step(): AutogradStep | null {
    const next = this.nextStep;
    if (!next) return null;

    if (next.pass === "forward") {
      for (const id of next.ids) {
        this.evaluate(this.nodes.get(id)!);
        this.computedIds.add(id);
      }
    } else {
      if (!this.backwardSeeded) this.startBackward();
      for (const id of next.ids) {
        this.propagate(this.nodes.get(id)!);
        this.backwardIds.add(id);
      }
    }
    this.last = next;
    return next;
  }

  toSnapshot(): AutogradSnapshot {
    return {
      nodes: [...this.nodes.values()].map((n) => ({ id: n.id, op: n.op, label: n.label, value: n.value })),
      edges: [...this.edges.values()].map((e) => ({ id: e.id, source: e.source, target: e.target, slot: e.slot })),
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
    // A snapshot from before operations had ports lists a target's edges in
    // port order, so they fill its ports in turn.
    const nextPort = new Map<string, number>();
    for (const def of snapshot.edges) {
      const slot = def.slot ?? nextPort.get(def.target) ?? 0;
      const edge = new AutogradEdge(def.id, def.source, def.target, slot);
      graph.edges.set(def.id, edge);
      nextPort.set(def.target, slot + 1);
      const num = Number.parseInt(def.id, 10);
      if (!Number.isNaN(num)) maxId = Math.max(maxId, num);
    }
    graph.outputId = snapshot.outputId;
    resetAutogradIdCounter(maxId + 1);
    graph.recomputeTopoOrder();
    return graph;
  }
}
