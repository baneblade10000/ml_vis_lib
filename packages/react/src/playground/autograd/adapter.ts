import type { Edge, Node } from "@xyflow/react";
import { OP_SPECS, type AutogradGraph, type AutogradOp } from "@ml-vis/core";

export const AUTOGRAD_DRAG_TYPE = "application/reactflow-autograd";

export const AUTOGRAD_NODE_WIDTH = 96;
export const AUTOGRAD_NODE_HEIGHT = 64;

export type AutogradNodeData = {
  op: AutogradOp;
  symbol: string;
  label: string;
  value: number;
  grad: number;
  isLeaf: boolean;
  isOutput: boolean;
  showGrad: boolean;
  selected: boolean;
};

export type AutogradEdgeData = {
  localDer: number;
  showGrad: boolean;
};

/** Diverging color for a gradient magnitude: crimson (+) → grey (0) → indigo (−). */
export function gradColor(grad: number): string {
  if (!Number.isFinite(grad) || grad === 0) return "#94a3b8";
  const mag = Math.min(1, Math.abs(grad) / 4);
  return grad > 0
    ? `rgba(220, 38, 58, ${0.35 + mag * 0.55})`
    : `rgba(79, 70, 229, ${0.35 + mag * 0.55})`;
}

export function autogradToFlow(
  graph: AutogradGraph,
  options: { selectedNodeId: string | null; selectedEdgeId: string | null; showGrad: boolean },
): { nodes: Node<AutogradNodeData>[]; edges: Edge<AutogradEdgeData>[] } {
  const nodes: Node<AutogradNodeData>[] = [];
  const edges: Edge<AutogradEdgeData>[] = [];

  for (const node of graph.nodes.values()) {
    const pos = graph.positions.get(node.id) ?? { x: 0, y: 0 };
    const spec = OP_SPECS[node.op];
    const isOutput = graph.outputId === node.id;
    // Note: React Flow reserves the "output" node type and paints its own frame,
    // so the sink node uses a custom "sink" type instead.
    nodes.push({
      id: node.id,
      type: node.isLeaf ? "leaf" : isOutput ? "sink" : "op",
      position: pos,
      width: AUTOGRAD_NODE_WIDTH,
      height: AUTOGRAD_NODE_HEIGHT,
      data: {
        op: node.op,
        symbol: spec.symbol,
        label: node.label ?? spec.label,
        value: node.value,
        grad: node.grad,
        isLeaf: node.isLeaf,
        isOutput,
        showGrad: options.showGrad,
        selected: options.selectedNodeId === node.id,
      },
    });
  }

  for (const edge of graph.edges.values()) {
    edges.push({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: "autograd",
      selected: options.selectedEdgeId === edge.id,
      data: { localDer: edge.localDer, showGrad: options.showGrad },
    });
  }

  return { nodes, edges };
}

export function autogradKindFromDrag(kind: string): AutogradOp | null {
  return kind in OP_SPECS ? (kind as AutogradOp) : null;
}
