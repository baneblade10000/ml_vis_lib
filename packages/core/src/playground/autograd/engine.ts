import { AutogradGraph } from "./graph";
import { AUTOGRAD_PRESETS, buildAutogradPreset, type AutogradPresetId } from "./presets";
import { OP_SPECS, type AutogradOp, type AutogradPosition } from "./types";

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
    this.recompute();
    return node.id;
  }

  connectNodes(sourceId: string, targetId: string): boolean {
    const edge = this.graph.connect(sourceId, targetId);
    if (!edge) return false;
    this.recompute();
    return true;
  }

  disconnectNodes(sourceId: string, targetId: string): boolean {
    const edge = [...this.graph.edges.values()].find((e) => e.source === sourceId && e.target === targetId);
    if (!edge) return false;
    this.graph.removeEdge(edge.id);
    this.recompute();
    return true;
  }

  removeEdge(edgeId: string): boolean {
    if (!this.graph.edges.has(edgeId)) return false;
    this.graph.removeEdge(edgeId);
    this.recompute();
    return true;
  }

  removeNode(nodeId: string): boolean {
    if (!this.graph.nodes.has(nodeId)) return false;
    this.graph.removeNode(nodeId);
    this.recompute();
    return true;
  }

  setNodePosition(nodeId: string, position: AutogradPosition): void {
    this.graph.setPosition(nodeId, position);
  }

  /** Update an editable leaf (input / const) scalar and refresh forward + backward. */
  setNodeValue(nodeId: string, value: number): boolean {
    const node = this.graph.getNode(nodeId);
    if (!node || !node.isLeaf || !Number.isFinite(value)) return false;
    node.value = value;
    this.recompute();
    return true;
  }

  setOutput(nodeId: string): void {
    this.graph.setOutput(nodeId);
    this.recompute();
  }

  loadPreset(preset: AutogradPresetId): void {
    this.config.preset = preset;
    this.graph = buildAutogradPreset(preset);
  }

  /**
   * Refresh topology + forward values after a structural edit. Gradients are
   * intentionally NOT recomputed here: the UI reveals them only when the user
   * runs the backward pass, so a graph edit leaves any shown gradients stale.
   */
  recompute(): void {
    this.graph.recomputeTopoOrder();
    this.graph.forward();
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
