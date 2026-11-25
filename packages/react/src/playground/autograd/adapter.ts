import type { Edge, Node } from "@xyflow/react";
import { OP_SPECS, type AutogradGraph, type AutogradOp, type AutogradPass } from "@ml-vis/core/autograd";
import { weightColor } from "@ml-vis/core/network";

export const AUTOGRAD_DRAG_TYPE = "application/reactflow-autograd";

export const AUTOGRAD_NODE_WIDTH = 96;
export const AUTOGRAD_NODE_HEIGHT = 64;

/** A node's role in the step-by-step passes: what the last step worked on, or what the next one will. */
export type AutogradStepRole = {
  state: "active" | "next";
  pass: AutogradPass;
};

export type AutogradNodeData = {
  op: AutogradOp;
  symbol: string;
  label: string;
  value: number;
  grad: number;
  isLeaf: boolean;
  isOutput: boolean;
  /** Whether the node's gradient is shown; it becomes known as the backward pass reaches the node. */
  showGrad: boolean;
  /** Whether the node's value is known; operations get it as the forward pass reaches them. */
  computed: boolean;
  step: AutogradStepRole | null;
  selected: boolean;
};

export type AutogradEdgeData = {
  localDer: number;
  /** Whether the local derivative is known; the backward pass sets it when it goes through the edge's target. */
  hasLocalDer: boolean;
  /** Whether the edge feeds a node the last step worked on: a forward step read its value, a backward step pushed a gradient back along it. */
  stepInput: boolean;
};

/** Smallest blend toward the endpoint: a gradient label has to stay readable. */
const GRAD_INK_FLOOR = 0.35;
/** |∂| at which the label reaches the scale endpoint. */
const GRAD_SATURATION = 4;

/**
 * Text color for a gradient on the numeric scale (negative → white at 0 →
 * positive), linearly in |grad|. No gradient takes the muted ink: white, the
 * scale's zero, would be invisible on a node.
 */
export function gradColor(grad: number): string {
  if (!Number.isFinite(grad) || grad === 0) return "#94a3b8";
  const mag = Math.min(1, Math.abs(grad) / GRAD_SATURATION);
  return weightColor(Math.sign(grad) * (GRAD_INK_FLOOR + (1 - GRAD_INK_FLOOR) * mag), 1);
}

export function autogradToFlow(
  graph: AutogradGraph,
  options: {
    selectedNodeId: string | null;
    selectedEdgeId: string | null;
  },
): { nodes: Node<AutogradNodeData>[]; edges: Edge<AutogradEdgeData>[] } {
  const nodes: Node<AutogradNodeData>[] = [];
  const edges: Edge<AutogradEdgeData>[] = [];
  const lastStep = graph.lastStep;
  const nextStep = graph.nextStep;
  const lastIds = new Set<string>(lastStep?.ids);
  const nextIds = new Set<string>(nextStep?.ids);
  const stepOf = (id: string): AutogradStepRole | null => {
    if (lastStep && lastIds.has(id)) return { state: "active", pass: lastStep.pass };
    if (nextStep && nextIds.has(id)) return { state: "next", pass: nextStep.pass };
    return null;
  };

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
        showGrad: graph.hasGrad(node.id),
        computed: graph.isComputed(node.id),
        step: stepOf(node.id),
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
      data: {
        localDer: edge.localDer,
        hasLocalDer: graph.hasLocalDer(edge.id),
        stepInput: lastIds.has(edge.target),
      },
    });
  }

  return { nodes, edges };
}

export function autogradKindFromDrag(kind: string): AutogradOp | null {
  return kind in OP_SPECS ? (kind as AutogradOp) : null;
}
