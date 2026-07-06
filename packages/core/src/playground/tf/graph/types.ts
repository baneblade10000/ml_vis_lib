import type { ActivationFunction } from "../nn";

export type GraphNodeKind = "input" | "dense" | "sum" | "output";

export interface GraphPosition {
  x: number;
  y: number;
}

export interface GraphNodeDef {
  id: string;
  kind: GraphNodeKind;
  activation: ActivationFunction;
  bias: number;
  label?: string;
}

export interface GraphEdgeDef {
  id: string;
  source: string;
  target: string;
  weight: number;
}

export interface GraphSnapshot {
  nodes: GraphNodeDef[];
  edges: GraphEdgeDef[];
  inputIds: string[];
  outputId: string;
  positions: Record<string, GraphPosition>;
}

export type ArchitecturePresetId = "mlp" | "resblock" | "mini-resnet" | "custom";

export interface PaletteNodeKind {
  kind: GraphNodeKind;
  label: string;
  description: string;
}

export const PALETTE_NODE_KINDS: PaletteNodeKind[] = [
  { kind: "dense", label: "Dense", description: "Weighted sum + activation" },
  { kind: "sum", label: "Add", description: "Sum inputs (residual merge)" },
  { kind: "output", label: "Output", description: "Network output node" },
];
