import { AutogradGraph } from "./graph";
import { AUTOGRAD_PRESETS, buildAutogradPreset, type AutogradPresetId } from "./presets";
import { OP_SPECS, type AutogradOp, type AutogradPosition, type AutogradStep } from "./types";

export interface CompGraphConfig {
  preset: AutogradPresetId;
}

export const DEFAULT_COMPGRAPH_CONFIG: CompGraphConfig = {
  preset: "expr",
};

/**
 * UI-facing wrapper around {@link AutogradGraph}, mirroring the role of
 * `PlaygroundEngine` for the neural-network layout. Owns the current graph,
 * runs forward/backward on demand, and exposes editing operations used by the
 * builder UI.
 *
 * Values and gradients become known only as a pass computes them: none on load
 * or after an edit, every node that is ready per {@link CompGraphEngine.runStep}
 * (independent ones advance together), all of them on
 * {@link CompGraphEngine.runForward} and {@link CompGraphEngine.runBackward}.
 * `runStep` goes through the forward pass first and, once that is complete,
 * through the backward pass.
 */
export class CompGraphEngine {
  config: CompGraphConfig;
  graph: AutogradGraph;
  private initialPreset: AutogradPresetId;

  constructor(config?: Partial<CompGraphConfig>) {
    this.config = { ...DEFAULT_COMPGRAPH_CONFIG, ...config };
    this.initialPreset = this.config.preset;
    this.graph = buildAutogradPreset(this.config.preset);
  }

  /** Add a palette node at the given canvas position. Leaf nodes get a default value. */
  addPaletteNode(op: AutogradOp, position: AutogradPosition): string {
    const spec = OP_SPECS[op];
    const value = op === "const" ? 1 : 0;
    const label = spec.leaf ? undefined : spec.label;
    const node = this.graph.addNode(op, { position, value, label });
    this.invalidate();
    return node.id;
  }

  connectNodes(sourceId: string, targetId: string): boolean {
    const edge = this.graph.connect(sourceId, targetId);
    if (!edge) return false;
    this.invalidate();
    return true;
  }

  disconnectNodes(sourceId: string, targetId: string): boolean {
    const edge = [...this.graph.edges.values()].find((e) => e.source === sourceId && e.target === targetId);
    if (!edge) return false;
    this.graph.removeEdge(edge.id);
    this.invalidate();
    return true;
  }

  removeEdge(edgeId: string): boolean {
    if (!this.graph.edges.has(edgeId)) return false;
    this.graph.removeEdge(edgeId);
    this.invalidate();
    return true;
  }

  removeNode(nodeId: string): boolean {
    if (!this.graph.nodes.has(nodeId)) return false;
    this.graph.removeNode(nodeId);
    this.invalidate();
    return true;
  }

  setNodePosition(nodeId: string, position: AutogradPosition): void {
    this.graph.setPosition(nodeId, position);
  }

  /** Update an editable leaf (input / const) scalar; every computed value and gradient goes stale with it. */
  setNodeValue(nodeId: string, value: number): boolean {
    const node = this.graph.getNode(nodeId);
    if (!node || !node.isLeaf || !Number.isFinite(value)) return false;
    node.value = value;
    this.invalidate();
    return true;
  }

  setOutput(nodeId: string): void {
    this.graph.setOutput(nodeId);
    this.invalidate();
  }

  loadPreset(preset: AutogradPresetId): void {
    this.config.preset = preset;
    this.graph = buildAutogradPreset(preset);
  }

  /**
   * Refresh topology after a graph edit and drop every computed value and
   * gradient: nothing is known about the edited graph until the user runs a
   * pass.
   */
  invalidate(): void {
    this.graph.recomputeTopoOrder();
    this.graph.resetPasses();
  }

  /** Run one step of whichever pass is up next; returns it, or null once the forward and backward passes are complete. */
  runStep(): AutogradStep | null {
    return this.graph.step();
  }

  runForward(): void {
    this.graph.recomputeTopoOrder();
    this.graph.forward();
  }

  runBackward(): void {
    this.graph.recomputeTopoOrder();
    this.graph.forward();
    this.graph.backward();
  }

  reset(): void {
    this.graph = buildAutogradPreset(this.config.preset);
  }

  resetToInitial(): void {
    this.config = { ...DEFAULT_COMPGRAPH_CONFIG, preset: this.initialPreset };
    this.graph = buildAutogradPreset(this.initialPreset);
  }
}

export { AUTOGRAD_PRESETS };
export type { AutogradPresetId };
