import type { Edge, Node } from "@xyflow/react";
import { MarkerType } from "@xyflow/react";
import { INPUTS, MLP_NODE_SIZE, MLP_OUTPUT_NODE_SIZE, mlpColumnXsFromCounts, weightColor, weightMagnitude, weightValueNormalized, type ComputationalGraph, type GraphNodeKind } from "@ml-vis/core/network";

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
  selectedNodeId: string | null;
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

/** Square cell size shared by every weight matrix on the canvas. */
export const MATRIX_CELL_PX = 28;
/** Extra horizontal room around a matrix inside its layer gap. */
const MATRIX_PAD = 32;
/** Floor so a 1-column matrix does not glue neighboring layers. */
const MATRIX_MIN_GAP = 72;

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

function nodeOuterSize(graph: ComputationalGraph, id: string): { w: number; h: number } {
  const isOut = graph.nodes.get(id)?.kind === "output";
  return isOut
    ? { w: OUTPUT_NODE_WIDTH, h: OUTPUT_NODE_HEIGHT }
    : { w: NODE_WIDTH, h: NODE_HEIGHT };
}

type LayerNodes = { id: string }[][];

/** Per-column X so each weight matrix fits at {@link MATRIX_CELL_PX}. */
function matrixColumnXs(graph: ComputationalGraph, layers: LayerNodes): number[] {
  const firstXs = (layers[0] ?? [])
    .map((n) => graph.positions.get(n.id)?.x)
    .filter((x): x is number => x != null);
  const xs: number[] = [firstXs.length ? Math.min(...firstXs) : 72];
  for (let i = 0; i < layers.length - 1; i++) {
    const layer = layers[i]!;
    const srcMaxW = Math.max(NODE_WIDTH, ...layer.map((n) => nodeOuterSize(graph, n.id).w));
    const matrixW = layer.length * MATRIX_CELL_PX + 2;
    const gap = Math.max(MATRIX_MIN_GAP, matrixW + MATRIX_PAD);
    xs.push(xs[i]! + srcMaxW + gap);
  }
  return xs;
}

function applyColumnXs(
  graph: ComputationalGraph,
  layers: LayerNodes,
  nodes: Node<NetworkNodeData>[],
  colXs: number[],
): Map<string, { x: number; y: number }> {
  const idToCol = new Map<string, number>();
  layers.forEach((layer, i) => {
    for (const n of layer) idToCol.set(n.id, i);
  });
  const positions = new Map<string, { x: number; y: number }>();
  for (const node of nodes) {
    const stored = graph.positions.get(node.id) ?? node.position;
    const col = idToCol.get(node.id);
    const pos = col === undefined ? stored : { x: colXs[col]!, y: stored.y };
    positions.set(node.id, pos);
    node.position = pos;
  }
  return positions;
}

function applyMatrixColumnLayout(
  graph: ComputationalGraph,
  layers: LayerNodes,
  nodes: Node<NetworkNodeData>[],
): Map<string, { x: number; y: number }> {
  return applyColumnXs(graph, layers, nodes, matrixColumnXs(graph, layers));
}

/** True when each layer is a vertical stack (same x), not a freeform graph. */
function layersAreColumnar(graph: ComputationalGraph, layers: LayerNodes): boolean {
  for (const layer of layers) {
    const xs = layer
      .map((n) => graph.positions.get(n.id)?.x)
      .filter((x): x is number => x != null);
    if (xs.length < 2) continue;
    if (Math.max(...xs) - Math.min(...xs) > 8) return false;
  }
  return layers.some((layer) => layer.length > 0);
}

function applyGraphColumnLayout(
  graph: ComputationalGraph,
  layers: LayerNodes,
  nodes: Node<NetworkNodeData>[],
): void {
  const firstXs = (layers[0] ?? [])
    .map((n) => graph.positions.get(n.id)?.x)
    .filter((x): x is number => x != null);
  const originX = firstXs.length ? Math.min(...firstXs) : 72;
  applyColumnXs(graph, layers, nodes, mlpColumnXsFromCounts(layers.map((l) => l.length), originX));
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
    selectedNodeId: string | null;
    edgeVizMode: EdgeVizMode;
    learningRate: number;
    gradients: Map<string, number>;
    gradScale: number;
    positions: Map<string, { x: number; y: number }>;
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

  const cellPx = MATRIX_CELL_PX;
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

  const srcPositions = sourceIds.map((id) => options.positions.get(id) ?? graph.positions.get(id));
  const destPositions = destIds.map((id) => options.positions.get(id) ?? graph.positions.get(id));
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
        selectedNodeId: options.selectedNodeId,
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
    const positions = applyMatrixColumnLayout(graph, layers, nodes);
    for (let i = 0; i < layers.length - 1; i++) {
      const sourceIds = layers[i]!.map((n) => n.id);
      const destIds = layers[i + 1]!.map((n) => n.id);
      const matrixNode = buildWeightMatrixNode(graph, i, sourceIds, destIds, {
        enabledFeatures: options.enabledFeatures,
        selectedEdgeId: options.selectedEdgeId,
        selectedNodeId: options.selectedNodeId,
        edgeVizMode,
        learningRate,
        gradients,
        gradScale,
        positions,
      });
      if (matrixNode) nodes.push(matrixNode);
    }
    return { nodes, edges: [] };
  }

  const layers = graph.toLayeredNetwork();
  if (layersAreColumnar(graph, layers)) {
    applyGraphColumnLayout(graph, layers, nodes);
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
    const incident =
      !!options.selectedNodeId &&
      (link.source.id === options.selectedNodeId || link.dest.id === options.selectedNodeId);
    const dimmed = !!options.selectedNodeId && !incident;
    const edgeSelected = options.selectedEdgeId === link.id;
    const emphasized = incident || edgeSelected;
    edges.push({
      id: link.id,
      source: link.source.id,
      target: link.dest.id,
      type: "weight",
      selected: edgeSelected,
      zIndex: emphasized ? 2 : 0,
      data: {
        weight: link.weight,
        gradient,
        learningRate,
        active: sourceActive,
        vizMode: edgeVizMode,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: emphasized ? 18 : 16,
        height: emphasized ? 18 : 16,
        color: dimmed
          ? "rgba(148,163,184,0.18)"
          : sourceActive
            ? stroke
            : "rgba(148,163,184,0.25)",
      },
      style: {
        stroke,
        strokeWidth: emphasized ? Math.max(3.5, strokeWidth) : strokeWidth,
        strokeOpacity: dimmed ? 0.08 : emphasized ? 1 : strokeOpacity,
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
