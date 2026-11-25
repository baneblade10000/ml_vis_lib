import { Position, type Edge, type Node, type NodeHandle } from "@xyflow/react";
import { OP_SPECS, type AutogradGraph, type AutogradOp, type AutogradPass } from "@ml-vis/core/autograd";
import { weightColor } from "@ml-vis/core/network";

export const AUTOGRAD_DRAG_TYPE = "application/reactflow-autograd";

export const AUTOGRAD_NODE_WIDTH = 96;
export const AUTOGRAD_NODE_HEIGHT = 64;
/** Vertical distance between two neighbouring input ports of one node. */
const PORT_SPACING = 16;
/** A port's rendered box: the 9px `.ag-handle` square, ring included, centred on the node border. */
const PORT_SIZE = 9;

/** The React Flow id of a node's input port; the edge feeding it carries the same as its target handle. */
export function autogradPortHandle(port: number): string {
  return `in-${port}`;
}

/** The port a connection ended on, from its handle id; undefined when the handle is not an input port. */
export function autogradPortOf(handle?: string | null): number | undefined {
  if (!handle?.startsWith("in-")) return undefined;
  const port = Number(handle.slice(3));
  return Number.isInteger(port) ? port : undefined;
}

/** A node's height: the base one, or taller once its input ports need more room than that. */
export function autogradNodeHeight(inputPorts: number): number {
  return Math.max(AUTOGRAD_NODE_HEIGHT, (inputPorts + 1) * PORT_SPACING);
}

/** How far down a node an input port sits, as a share of its height: the ports are spread evenly, margins included. */
export function autogradPortOffset(inputPorts: number, port: number): number {
  return (port + 1) / (inputPorts + 1);
}

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
  /** One entry per input port of an operation, down its left side: whether an edge feeds it. A leaf has none. */
  inputPorts: boolean[];
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
    const type = node.isLeaf ? "leaf" : isOutput ? "sink" : "op";
    const fed = new Set(graph.inputEdges(node.id).map((edge) => edge.slot));
    const inputPorts = Array.from({ length: graph.inputPortCount(node.id) }, (_, port) => fed.has(port));
    const height = autogradNodeHeight(inputPorts.length);
    // A handle's box as React Flow measures it in the DOM, so that declaring it here gives the same bounds.
    const box = (x: number, y: number) => ({
      x: x - PORT_SIZE / 2,
      y: y - PORT_SIZE / 2,
      width: PORT_SIZE,
      height: PORT_SIZE,
    });
    const handles: NodeHandle[] = inputPorts.map((_, port) => ({
      type: "target",
      position: Position.Left,
      id: autogradPortHandle(port),
      ...box(0, autogradPortOffset(inputPorts.length, port) * height),
    }));
    if (type !== "sink") {
      handles.push({ type: "source", position: Position.Right, ...box(AUTOGRAD_NODE_WIDTH, height / 2) });
    }
    nodes.push({
      id: node.id,
      type,
      position: pos,
      width: AUTOGRAD_NODE_WIDTH,
      height,
      // Declared, not left to be measured: an edge and the port it feeds appear in the same frame.
      handles,
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
        inputPorts,
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
      targetHandle: autogradPortHandle(edge.slot),
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
