import type { Edge, Node } from "@xyflow/react";
import {
  INPUTS,
  weightColor,
  type ComputationalGraph,
  type GraphNodeKind,
} from "@ml-vis/core";

export const NODE_WIDTH = 52;
export const NODE_HEIGHT = 52;
export const OUTPUT_NODE_WIDTH = 168;
export const OUTPUT_NODE_HEIGHT = 168;

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
    edges.push({
      id: link.id,
      source: link.source.id,
      target: link.dest.id,
      type: "weight",
      selected: options.selectedEdgeId === link.id,
      data: { weight: link.weight, active: sourceActive },
      style: {
        stroke: weightColor(link.weight),
        strokeWidth: Math.max(2, Math.min(8, 2 + (Math.abs(link.weight) / 5) * 6)),
        strokeOpacity: sourceActive ? Math.max(0.45, Math.min(1, 0.45 + Math.abs(link.weight) * 0.55)) : 0.12,
      },
    });
  }

  return { nodes, edges };
}
