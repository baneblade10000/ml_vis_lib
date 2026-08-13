import type { Edge, Node } from "@xyflow/react";
import { MarkerType } from "@xyflow/react";
import { INPUTS, MLP_COL_SPACING, MLP_NODE_SIZE, MLP_OUTPUT_NODE_SIZE, weightColor, weightMagnitude, weightValueNormalized, type ComputationalGraph, type GraphNodeKind } from "@ml-vis/core/network";

// Single source of truth lives in @ml-vis/core (mlp-layout.ts); these re-export
// the canonical values so layout geometry and rendered node sizes can never drift.
export const NODE_WIDTH = MLP_NODE_SIZE;
export const NODE_HEIGHT = MLP_NODE_SIZE;
export const OUTPUT_NODE_WIDTH = MLP_OUTPUT_NODE_SIZE;
export const OUTPUT_NODE_HEIGHT = MLP_OUTPUT_NODE_SIZE;

export type DataPoint = { x: number; y: number; label: number };

/** How connections between layers are drawn on the canvas. */
export type LayoutVizMode = "graph" | "matrix";

export type WeightMatrixCellData = {
  linkId: string | null;
  weight: number;
  gradient: number;
  active: boolean;
};

export type WeightMatrixPayload = {
  sourceIds: string[];
  destIds: string[];
  sourceLabels: string[];
  destLabels: string[];
  cells: WeightMatrixCellData[][];
  selectedEdgeId: string | null;
  vizMode: EdgeVizMode;
  learningRate: number;
  cellPx: number;
  /** Global max|∂E/∂w| so gradient colors match edge viz across matrices. */
  gradScale: number;
};

export type NetworkNodeData = {
  kind: GraphNodeKind | "weightMatrix";
  label: string;
  bias?: number;
  active?: boolean;
  trainData?: DataPoint[];
  lossTest?: number;
  lossTrain?: number;
  discretize: boolean;
  selected: boolean;
  paintGeneration?: number;
  /** Present when `kind === "weightMatrix"`. */
  matrix?: WeightMatrixPayload;
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

const MATRIX_CELL_MIN = 8;
const MATRIX_CELL_MAX = 18;
/** Horizontal room between adjacent neuron columns for a weight matrix. */
const MATRIX_GAP = MLP_COL_SPACING - NODE_WIDTH;

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

function nodeDisplayLabel(graph: ComputationalGraph, id: string): string {
  if (id in INPUTS) return INPUTS[id]?.label ?? id;
  const node = graph.nodes.get(id);
  return node?.label ?? id;
}

function matrixCellPx(rows: number, cols: number): number {
  const dim = Math.max(rows, cols, 1);
  const budget = MATRIX_GAP - 20;
  const raw = Math.floor(budget / dim);
  return Math.max(MATRIX_CELL_MIN, Math.min(MATRIX_CELL_MAX, raw));
}

function nodeOuterSize(graph: ComputationalGraph, id: string): { w: number; h: number } {
  const isOut = graph.nodes.get(id)?.kind === "output";
  return isOut
    ? { w: OUTPUT_NODE_WIDTH, h: OUTPUT_NODE_HEIGHT }
    : { w: NODE_WIDTH, h: NODE_HEIGHT };
}

/** Build one weight-matrix node sitting in the gap between two layer columns. */
function buildWeightMatrixNode(
  graph: ComputationalGraph,
  layerIndex: number,
  sourceIds: string[],
  destIds: string[],
  options: {
    enabledFeatures: Record<string, boolean>;
    selectedEdgeId: string | null;
    edgeVizMode: EdgeVizMode;
    learningRate: number;
    gradients: Map<string, number>;
    gradScale: number;
  },
): Node<NetworkNodeData> | null {
  if (!sourceIds.length || !destIds.length) return null;

  const destSet = new Set(destIds);
  const linkByPair = new Map<string, { id: string; weight: number; isDead: boolean }>();
  for (const srcId of sourceIds) {
    const src = graph.nodes.get(srcId);
    if (!src) continue;
    for (const link of src.outputs) {
      if (destSet.has(link.dest.id)) {
        linkByPair.set(`${srcId}\0${link.dest.id}`, link);
      }
    }
  }

  const cellPx = matrixCellPx(destIds.length, sourceIds.length);
  const cells: WeightMatrixCellData[][] = destIds.map((destId) =>
    sourceIds.map((srcId) => {
      const link = linkByPair.get(`${srcId}\0${destId}`);
      const sourceActive =
        srcId in options.enabledFeatures ? options.enabledFeatures[srcId] : true;
      return {
        linkId: link?.id ?? null,
        weight: link?.weight ?? 0,
        gradient: link ? (options.gradients.get(link.id) ?? 0) : 0,
        active: !!link && sourceActive && !link.isDead,
      };
    }),
  );

  const matrixW = sourceIds.length * cellPx + 2;
  const matrixH = destIds.length * cellPx + 2;

  const srcPositions = sourceIds.map((id) => graph.positions.get(id));
  const destPositions = destIds.map((id) => graph.positions.get(id));
  if (srcPositions.some((p) => !p) || destPositions.some((p) => !p)) return null;

  const srcRight = Math.max(
    ...sourceIds.map((id, i) => srcPositions[i]!.x + nodeOuterSize(graph, id).w),
  );
  const destLeft = Math.min(...destPositions.map((p) => p!.x));

  const centers: number[] = [];
  sourceIds.forEach((id, i) => {
    centers.push(srcPositions[i]!.y + nodeOuterSize(graph, id).h / 2);
  });
  destIds.forEach((id, i) => {
    centers.push(destPositions[i]!.y + nodeOuterSize(graph, id).h / 2);
  });
  const centerY = centers.reduce((a, b) => a + b, 0) / centers.length;

  return {
    id: `__weight_matrix_${layerIndex}`,
    type: "weightMatrix",
    position: {
      x: (srcRight + destLeft) / 2 - matrixW / 2,
      y: centerY - matrixH / 2,
    },
    width: matrixW,
    height: matrixH,
    draggable: false,
    connectable: false,
    selectable: false,
    focusable: false,
    data: {
      kind: "weightMatrix",
      label: `W${layerIndex}`,
      discretize: false,
      selected: false,
      matrix: {
        sourceIds,
        destIds,
        sourceLabels: sourceIds.map((id) => nodeDisplayLabel(graph, id)),
        destLabels: destIds.map((id) => nodeDisplayLabel(graph, id)),
        cells,
        selectedEdgeId: options.selectedEdgeId,
        vizMode: options.edgeVizMode,
        learningRate: options.learningRate,
        cellPx,
        gradScale: options.gradScale,
      },
    },
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
    layoutVizMode?: LayoutVizMode;
  },
): { nodes: Node<NetworkNodeData>[]; edges: Edge<WeightEdgeData>[] } {
  const nodes: Node<NetworkNodeData>[] = [];
  const edges: Edge<WeightEdgeData>[] = [];
  const edgeVizMode = options.edgeVizMode ?? "weight";
  const learningRate = options.learningRate ?? 0;
  const layoutVizMode = options.layoutVizMode ?? "graph";

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

  if (layoutVizMode === "matrix") {
    const layers = graph.toLayeredNetwork();
    for (let i = 0; i < layers.length - 1; i++) {
      const sourceIds = layers[i]!.map((n) => n.id);
      const destIds = layers[i + 1]!.map((n) => n.id);
      const matrixNode = buildWeightMatrixNode(graph, i, sourceIds, destIds, {
        enabledFeatures: options.enabledFeatures,
        selectedEdgeId: options.selectedEdgeId,
        edgeVizMode,
        learningRate,
        gradients,
        gradScale,
      });
      if (matrixNode) nodes.push(matrixNode);
    }
    return { nodes, edges: [] };
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
