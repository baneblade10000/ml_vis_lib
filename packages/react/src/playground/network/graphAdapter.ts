import type { Edge, Node } from "@xyflow/react";
import { MarkerType } from "@xyflow/react";
import {
  INPUTS,
  MLP_NODE_SIZE,
  MLP_OUTPUT_NODE_SIZE,
  weightColor,
  weightMagnitude,
  weightValueNormalized,
  type ComputationalGraph,
  type GraphNodeKind,
} from "@ml-vis/core";

// Single source of truth lives in @ml-vis/core (mlp-layout.ts); these re-export
// the canonical values so layout geometry and rendered node sizes can never drift.
export const NODE_WIDTH = MLP_NODE_SIZE;
export const NODE_HEIGHT = MLP_NODE_SIZE;
export const OUTPUT_NODE_WIDTH = MLP_OUTPUT_NODE_SIZE;
export const OUTPUT_NODE_HEIGHT = MLP_OUTPUT_NODE_SIZE;

export type DataPoint = { x: number; y: number; label: number };

export type NetworkNodeData = {
  kind: GraphNodeKind;
  label: string;
  bias?: number;
  active?: boolean;
  trainData?: DataPoint[];
  lossTest?: number;
  lossTrain?: number;
  discretize: boolean;
  selected: boolean;
  paintGeneration?: number;
};

export type WeightEdgeData = {
  weight: number;
  active: boolean;
};

export function graphToFlow(
  graph: ComputationalGraph,
  options: {
    enabledFeatures: Record<string, boolean>;
    discretize: boolean;
    selectedNodeId: string | null;
    selectedEdgeId: string | null;
    trainData?: DataPoint[];
    lossTest?: number;
    lossTrain?: number;
    paintGeneration?: number;
  },
): { nodes: Node<NetworkNodeData>[]; edges: Edge<WeightEdgeData>[] } {
  const nodes: Node<NetworkNodeData>[] = [];
  const edges: Edge<WeightEdgeData>[] = [];

  for (const id of graph.inputIds) {
    const pos = graph.positions.get(id) ?? { x: 0, y: 0 };
    nodes.push({
      id,
      type: "feature",
      position: pos,
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      data: {
        kind: "input",
        label: INPUTS[id]?.label ?? id,
        active: options.enabledFeatures[id],
        discretize: options.discretize,
        selected: options.selectedNodeId === id,
        paintGeneration: options.paintGeneration,
      },
    });
  }

  for (const [id, node] of graph.nodes) {
    if (node.kind === "input") continue;
    const pos = graph.positions.get(id) ?? { x: 0, y: 0 };
    // "readout" instead of "output": React Flow reserves the "output" node type
    // and applies its built-in chrome (border + padding) around custom nodes.
    const type = node.kind === "sum" ? "sum" : node.kind === "output" ? "readout" : "dense";
    const isOutput = node.kind === "output";
    nodes.push({
      id,
      type,
      position: pos,
      width: isOutput ? OUTPUT_NODE_WIDTH : NODE_WIDTH,
      height: isOutput ? OUTPUT_NODE_HEIGHT : NODE_HEIGHT,
      data: {
        kind: node.kind,
        label: node.label ?? node.kind,
        bias: node.bias,
        trainData: isOutput ? options.trainData : undefined,
        lossTest: isOutput ? options.lossTest : undefined,
        lossTrain: isOutput ? options.lossTrain : undefined,
        discretize: options.discretize,
        selected: options.selectedNodeId === id,
        paintGeneration: options.paintGeneration,
      },
    });
  }

  for (const link of graph.getAllLinks()) {
    const sourceActive =
      link.source.id in options.enabledFeatures ? options.enabledFeatures[link.source.id] : true;
    // One normalized scale (tanh) drives color, width and opacity together so a
    // large weight saturates all three at the same point instead of clamping at
    // different thresholds.
    const mag = weightMagnitude(link.weight);
    const stroke = weightColor(weightValueNormalized(link.weight));
    edges.push({
      id: link.id,
      source: link.source.id,
      target: link.dest.id,
      type: "weight",
      selected: options.selectedEdgeId === link.id,
      data: { weight: link.weight, active: sourceActive },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 16,
        height: 16,
        // Color the marker to match the edge stroke; react-flow keys the
        // marker by the color string, so this is cheap.
        color: sourceActive ? stroke : "rgba(148,163,184,0.25)",
      },
      style: {
        stroke,
        // 2px floor for faint weights, ~7.5px ceiling once |tanh(w)| → 1.
        strokeWidth: 2 + mag * 5.5,
        strokeOpacity: sourceActive ? 0.45 + mag * 0.55 : 0.12,
      },
    });
  }

  return { nodes, edges };
}

export function flowKindFromDrag(kind: string): GraphNodeKind | null {
  if (kind === "dense") return kind;
  return null;
}

export const PALETTE_DRAG_TYPE = "application/reactflow-network";
