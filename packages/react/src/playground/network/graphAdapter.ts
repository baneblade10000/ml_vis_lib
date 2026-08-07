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

/** What drives edge stroke width / color / opacity. */
export type EdgeVizMode = "weight" | "gradient";

export type WeightEdgeData = {
  weight: number;
  /** ∂E/∂w for this link (batch mean when accumulating, else last sample). */
  gradient: number;
  /** Current learning rate — used to show Δw = −lr·∂ on hover. */
  learningRate: number;
  active: boolean;
  vizMode: EdgeVizMode;
};

/** Current ∂E/∂w used for visualization (batch mean → last batch → last sample). */
export function linkPartialDerivative(link: {
  errorDer: number;
  accErrorDer: number;
  numAccumulatedDers: number;
  lastGradient?: number;
}): number {
  if (link.numAccumulatedDers > 0) {
    return link.accErrorDer / link.numAccumulatedDers;
  }
  // After a batch update accumulators are cleared; keep the last mean ∂E/∂w.
  return link.lastGradient ?? link.errorDer;
}

function edgeStrokeStyle(
  value: number,
  active: boolean,
  opts?: { alreadyNormalized?: boolean },
): {
  stroke: string;
  strokeWidth: number;
  strokeOpacity: number;
} {
  // Weights use tanh so small values stay legible and large ones saturate.
  // Gradients are already scaled by max|∂E/∂w|, so map linearly in [-1, 1].
  const mag = opts?.alreadyNormalized
    ? Math.min(1, Math.abs(value))
    : weightMagnitude(value);
  const stroke = weightColor(
    opts?.alreadyNormalized ? Math.max(-1, Math.min(1, value)) : weightValueNormalized(value),
  );
  return {
    stroke,
    // 2px floor for faint values, ~7.5px ceiling at full magnitude.
    strokeWidth: 2 + mag * 5.5,
    strokeOpacity: active ? 0.45 + mag * 0.55 : 0.12,
  };
}

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
    edgeVizMode?: EdgeVizMode;
    learningRate?: number;
  },
): { nodes: Node<NetworkNodeData>[]; edges: Edge<WeightEdgeData>[] } {
  const nodes: Node<NetworkNodeData>[] = [];
  const edges: Edge<WeightEdgeData>[] = [];
  const edgeVizMode = options.edgeVizMode ?? "weight";
  const learningRate = options.learningRate ?? 0;

  // Relative scale so the largest |∂E/∂w| saturates. sqrt keeps earlier-layer
  // grads visible in deep nets where the output layer would otherwise dominate.
  const links = graph.getAllLinks();
  let maxAbsGrad = 0;
  const gradients = new Map<string, number>();
  for (const link of links) {
    const g = linkPartialDerivative(link);
    gradients.set(link.id, g);
    maxAbsGrad = Math.max(maxAbsGrad, Math.abs(g));
  }
  const gradScale = maxAbsGrad > 1e-12 ? maxAbsGrad : 1;

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

  for (const link of links) {
    const sourceActive =
      link.source.id in options.enabledFeatures ? options.enabledFeatures[link.source.id] : true;
    const gradient = gradients.get(link.id) ?? 0;
    const isGradient = edgeVizMode === "gradient";
    let vizValue = link.weight;
    if (isGradient) {
      const ratio = Math.abs(gradient) / gradScale;
      vizValue = Math.sign(gradient) * Math.sqrt(ratio);
    }
    const { stroke, strokeWidth, strokeOpacity } = edgeStrokeStyle(vizValue, sourceActive, {
      alreadyNormalized: isGradient,
    });
    edges.push({
      id: link.id,
      source: link.source.id,
      target: link.dest.id,
      type: "weight",
      selected: options.selectedEdgeId === link.id,
      data: {
        weight: link.weight,
        gradient,
        learningRate,
        active: sourceActive,
        vizMode: edgeVizMode,
      },
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
        strokeWidth,
        strokeOpacity,
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
