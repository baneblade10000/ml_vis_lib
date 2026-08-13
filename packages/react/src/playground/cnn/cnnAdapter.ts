import type { Edge, Node } from "@xyflow/react";
import { weightColorZeroWhite, weightMagnitude, weightValueNormalized } from "@ml-vis/core/network";
import { type CnnLayerView, type CnnMode, type FeatureMapSnapshot, type LayerKind, type LayerShape } from "@ml-vis/core/cnn";
import type { CnnMessages } from "./messages";

/** Minimal layer surface needed for localized titles. */
type LabelableLayer = { kind: LayerKind; label: () => string };

/** Pipeline view for React Flow (from train-worker snapshot or live engine). */
export type CnnPipelineView = {
  mode: CnnMode;
  layers: CnnLayerView[];
};

/** Pixel geometry for the CNN flow graph. */
export const CNN_COL_GAP = 72;
/** @deprecated use {@link CNN_COL_GAP}; kept for any external imports. */
export const CNN_COL_SPACING = CNN_COL_GAP;
export const CNN_ORIGIN_X = 80;
export const CNN_NODE_WIDTH = 150;

/** Cap displayed units so Flatten/Dense don't dominate the canvas. */
export const CNN_MAX_VIS_UNITS = 32;
/** Max pixel height of a Flatten/Dense unit stack — keeps cells readable as squares. */
export const CNN_MAX_STACK_HEIGHT = 420;

/**
 * On-screen size of one activation / weight cell.
 * Kernels and feature maps share this so a kernel pixel matches a map pixel.
 */
export const CNN_CELL_PX = 8;

const MAP_GAP = 3;
const PAIR_GAP = 4;
const BIAS_CHIP = 13; // bias square + gap beside kernel
const MAP_GRID_MAX_W = 140;
const NODE_CHROME = 38; // label + padding + gaps

export function cnnGridPx(
  rows: number,
  cols: number,
  cellPx = CNN_CELL_PX,
): { w: number; h: number } {
  return {
    w: Math.max(1, cols) * cellPx,
    h: Math.max(1, rows) * cellPx,
  };
}

/** Width of one conv channel cell: bias + kernel + gap + feature map. */
function convChannelCellW(mapCols: number, kernelCols: number): number {
  const map = cnnGridPx(1, mapCols);
  const ker = cnnGridPx(1, kernelCols);
  return BIAS_CHIP + ker.w + PAIR_GAP + map.w;
}

/** Evenly sample a 1-D series down to `target` samples. */
export function downsample1D(values: number[], target: number): number[] {
  if (values.length <= target) return values;
  const out = new Array<number>(target);
  for (let i = 0; i < target; i++) {
    const src = Math.floor(((i + 0.5) * values.length) / target);
    out[i] = values[Math.min(values.length - 1, src)]!;
  }
  return out;
}

/** Source indices used when downsampling a length-`n` vector to `target` slots. */
export function sampleIndices(n: number, target: number): number[] {
  const count = Math.min(Math.max(0, n), Math.max(0, target));
  if (n <= target) return Array.from({ length: count }, (_, i) => i);
  const out = new Array<number>(count);
  for (let i = 0; i < count; i++) {
    out[i] = Math.min(n - 1, Math.floor(((i + 0.5) * n) / target));
  }
  return out;
}

/** Stroke style matching NN weight edges (signed color + tanh magnitude). */
export function cnnWeightStroke(weight: number, active = true): {
  stroke: string;
  strokeWidth: number;
  strokeOpacity: number;
} {
  const mag = weightMagnitude(weight);
  return {
    stroke: weightColorZeroWhite(weightValueNormalized(weight)),
    strokeWidth: 2 + mag * 5.5,
    strokeOpacity: active ? 0.45 + mag * 0.55 : 0.12,
  };
}

/** Circle/square stack size shared by Flatten / Dense weight columns. */
export function unitStackSize(
  length: number,
  columns = 1,
): { width: number; height: number; visCount: number; d: number; gap: number; gapX: number } {
  const n = Math.max(1, length);
  const visCount = Math.min(n, CNN_MAX_VIS_UNITS);
  const gap = 3;
  const gapX = 6;
  // Cell diameter ~2× the previous scale (was max 10.5 / min floor 2.5).
  let d = Math.max(10.5, Math.min(21, (520 / visCount) * 3));
  let height = visCount * d + gap * Math.max(0, visCount - 1);
  if (height > CNN_MAX_STACK_HEIGHT) {
    d = Math.max(5, (CNN_MAX_STACK_HEIGHT - gap * Math.max(0, visCount - 1)) / visCount);
    height = visCount * d + gap * Math.max(0, visCount - 1);
  }
  return {
    width: Math.round(columns * d + gapX * Math.max(0, columns - 1)),
    height: Math.round(height),
    visCount,
    d,
    gap,
    gapX,
  };
}

function estimateNodeSize(
  kind: LayerKind,
  shape: LayerShape | undefined,
  mode: CnnMode,
  /** Incoming vector length for dense weight columns. */
  inputLength = 64,
  /** Conv kernel spatial size (k) when known from feature-map dump. */
  kernelDim = 3,
): { width: number; height: number } {
  if (kind === "flatten" || kind === "gap2d" || kind === "gap1d") {
    const n = shape?.kind === "1d" ? shape.length : 32;
    const stack = unitStackSize(n, 1);
    return { width: Math.max(48, stack.width + 28), height: NODE_CHROME + stack.height + 14 };
  }
  if (kind === "dense") {
    void inputLength;
    // One neuron column (weights are drawn as inter-layer edges, NN-style).
    const units = shape?.kind === "1d" ? shape.length : 1;
    const stack = unitStackSize(Math.max(1, units), 1);
    const cell = Math.max(stack.d, 28);
    const stackH = stack.visCount * cell + stack.gap * Math.max(0, stack.visCount - 1);
    return {
      width: Math.max(72, cell + 40),
      height: NODE_CHROME + stackH + 18,
    };
  }
  if (kind === "output") {
    return { width: 108, height: 168 };
  }

  const channels = Math.max(1, Math.min(16, shape?.channels ?? 1));
  const isConv = kind === "conv2d" || kind === "conv1d";
  const isPool = kind === "pool2d" || kind === "pool1d";
  const mapRows = shape?.kind === "2d" ? shape.rows : 1;
  const mapCols =
    shape?.kind === "2d" ? shape.cols : shape?.kind === "1d" ? shape.length : 16;
  const map = cnnGridPx(mapRows, Math.max(1, mapCols));
  const ker = cnnGridPx(kernelDim, kernelDim);

  if (mode === "1d" || shape?.kind === "1d") {
    const sigLen = shape?.kind === "1d" ? shape.length : 64;
    const sig = cnnGridPx(1, Math.max(1, sigLen));
    const ker1 = cnnGridPx(1, kernelDim);
    const rowH = isConv ? Math.max(8, ker1.h, 8) : 8;
    const gridH = channels * rowH + Math.max(0, channels - 1) * MAP_GAP;
    const width = isConv
      ? Math.max(CNN_NODE_WIDTH, BIAS_CHIP + ker1.w + PAIR_GAP + sig.w + 28)
      : CNN_NODE_WIDTH;
    return { width, height: NODE_CHROME + gridH };
  }
  if (isConv) {
    const cellW = convChannelCellW(mapCols, kernelDim);
    const rowH = Math.max(map.h, ker.h);
    const gridH = channels * rowH + Math.max(0, channels - 1) * MAP_GAP;
    return {
      width: Math.max(CNN_NODE_WIDTH, cellW + 28),
      height: NODE_CHROME + gridH,
    };
  }
  if (isPool) {
    const gridH = channels * map.h + Math.max(0, channels - 1) * MAP_GAP;
    return {
      width: Math.max(CNN_NODE_WIDTH, map.w + 28),
      height: NODE_CHROME + gridH,
    };
  }
  const tile = Math.max(map.w, map.h);
  const perRow = Math.max(1, Math.floor((MAP_GRID_MAX_W + MAP_GAP) / (tile + MAP_GAP)));
  const gridRows = Math.ceil(channels / perRow);
  const gridH = gridRows * tile + Math.max(0, gridRows - 1) * MAP_GAP;
  return { width: CNN_NODE_WIDTH, height: NODE_CHROME + gridH };
}

/** Localize engine labels for the graph chrome. */
export function formatCnnNodeLabel(layer: LabelableLayer, t: CnnMessages): string {
  const raw = layer.label();
  switch (layer.kind) {
    case "input":
      return raw.replace(/^Input/, t.input);
    case "conv2d":
      return raw.replace(/^Conv2D/, t.paletteConv);
    case "conv1d":
      return raw.replace(/^Conv1D/, t.paletteConv);
    case "pool2d":
    case "pool1d":
      return raw
        .replace(/^Max Pool/, `${t.poolMax} ${t.palettePool}`)
        .replace(/^Avg Pool/, `${t.poolAvg} ${t.palettePool}`);
    case "gap2d":
    case "gap1d":
      return t.paletteGap;
    case "flatten":
      return t.flatten;
    case "dense":
      return raw.replace(/^Dense/, t.paletteDense);
    case "output":
      return t.output;
    default:
      return raw;
  }
}

/** Reactive payload carried by each React Flow node. */
export type CnnNodeData = {
  layerId: string;
  kind: LayerKind;
  label: string;
  mode: CnnMode;
  /** Channel count to render (feature maps / kernels). */
  channels: number;
  /** Spatial dims for 2-D layers (feature map size). */
  rows?: number;
  cols?: number;
  /** Length for 1-D layers. */
  length?: number;
  /** Conv kernel spatial size used for width estimates / layout invalidation. */
  kernelSize?: number;
  /** Param count of the layer. */
  params: number;
  /** Aggregate weight magnitude (tanh) for the connecting edge, or null. */
  weightMag: number | null;
  /** Output node: current loss. */
  loss?: number;
  probability?: number;
  selected: boolean;
  paintGeneration?: number;
};

export type CnnEdgeData = {
  weightMag: number;
  active: boolean;
  /** Signed weight when this is a per-unit dense/output edge (NN-style). */
  weight?: number;
};

/** React Flow node-type name per layer kind (avoiding React Flow's reserved "output"). */
function flowTypeFor(kind: LayerKind): string {
  switch (kind) {
    case "input":
      return "cnnInput";
    case "conv2d":
    case "conv1d":
      return "cnnConv";
    case "pool2d":
    case "pool1d":
      return "cnnPool";
    case "flatten":
    case "gap2d":
    case "gap1d":
      return "cnnFlatten";
    case "dense":
      return "cnnDense";
    case "output":
      return "cnnReadout";
  }
}

function channelCount(shape: LayerShape | undefined): number {
  if (!shape) return 1;
  return shape.channels;
}

export function cnnPipelineToFlow(
  pipeline: CnnPipelineView,
  options: {
    selectedNodeId: string | null;
    paintGeneration?: number;
    featureMaps: FeatureMapSnapshot[];
    /** Preferred kernel sizes (from config) — drives width before snapshots catch up. */
    kernelSizeByLayerId?: Record<string, number>;
    loss?: number;
    probability?: number;
    /** Localized node titles; defaults to layer.label string. */
    labelFor?: (layer: LabelableLayer) => string;
  },
): { nodes: Node<CnnNodeData>[]; edges: Edge<CnnEdgeData>[] } {
  const layers = pipeline.layers;
  const nodes: Node<CnnNodeData>[] = [];
  const edges: Edge<CnnEdgeData>[] = [];
  const mode = pipeline.mode;
  const labelFor =
    options.labelFor ??
    ((layer: LabelableLayer) => layer.label());

  const sizes = layers.map((layer, idx) => {
    const shape = layer.shape;
    const prev = layers[idx - 1]?.shape;
    const inputLength =
      prev?.kind === "1d" ? prev.length : prev?.kind === "2d" ? prev.rows * prev.cols * prev.channels : 64;
    const snap = options.featureMaps.find((m) => m.layerId === layer.id);
    const kernelDim =
      options.kernelSizeByLayerId?.[layer.id] ??
      snap?.kernels2d?.[0]?.length ??
      snap?.kernels1d?.[0]?.length ??
      3;
    return estimateNodeSize(layer.kind, shape, mode, inputLength, kernelDim);
  });
  const maxH = Math.max(1, ...sizes.map((s) => s.height));
  void maxH;

  // X packs left→right by full node width + gap (half-width step overlapped on 32×32).
  // Y is the vertical center (nodeOrigin [0, 0.5] in the graph).
  let x = CNN_ORIGIN_X;
  layers.forEach((layer, idx) => {
    const shape = layer.shape;
    const type = flowTypeFor(layer.kind);
    const isOutput = layer.kind === "output";
    const wMag = layer.weightMag;
    const { width, height } = sizes[idx]!;
    const snap = options.featureMaps.find((m) => m.layerId === layer.id);
    const kernelSize =
      options.kernelSizeByLayerId?.[layer.id] ??
      snap?.kernels2d?.[0]?.length ??
      snap?.kernels1d?.[0]?.length;
    const labelable: LabelableLayer = { kind: layer.kind, label: () => layer.label };
    nodes.push({
      id: layer.id,
      type,
      position: {
        x,
        y: 0,
      },
      width,
      height,
      data: {
        layerId: layer.id,
        kind: layer.kind,
        label: labelFor(labelable),
        mode,
        channels: channelCount(shape),
        rows: shape?.kind === "2d" ? shape.rows : undefined,
        cols: shape?.kind === "2d" ? shape.cols : undefined,
        length: shape?.kind === "1d" ? shape.length : undefined,
        kernelSize:
          layer.kind === "conv2d" || layer.kind === "conv1d" ? (kernelSize ?? 3) : undefined,
        params: layer.params,
        weightMag: wMag,
        loss: isOutput ? options.loss : undefined,
        probability: isOutput ? options.probability : undefined,
        selected: options.selectedNodeId === layer.id,
        paintGeneration: options.paintGeneration,
      },
    });
    if (idx > 0) {
      const prev = layers[idx - 1]!;
      const snap = options.featureMaps.find((m) => m.layerId === layer.id);
      const matrix = snap?.matrix;
      const prevLen =
        prev.shape?.kind === "1d"
          ? prev.shape.length
          : matrix?.[0]?.length ?? 0;
      const outLen =
        layer.shape?.kind === "1d" ? layer.shape.length : matrix?.length ?? 1;
      const canFanIn =
        matrix != null &&
        matrix.length > 0 &&
        (prev.kind === "flatten" ||
          prev.kind === "gap2d" ||
          prev.kind === "gap1d" ||
          prev.kind === "dense") &&
        (layer.kind === "dense" || layer.kind === "output");

      if (canFanIn) {
        const inIdx = sampleIndices(prevLen || matrix[0]!.length, CNN_MAX_VIS_UNITS);
        const outIdx = sampleIndices(outLen || matrix.length, CNN_MAX_VIS_UNITS);
        for (let oi = 0; oi < outIdx.length; oi++) {
          const o = outIdx[oi]!;
          const row = matrix[o] ?? matrix[0]!;
          for (let ii = 0; ii < inIdx.length; ii++) {
            const i = inIdx[ii]!;
            const w = row[i] ?? 0;
            const style = cnnWeightStroke(w, true);
            edges.push({
              id: `${prev.id}->${layer.id}:${ii}->${oi}`,
              source: prev.id,
              target: layer.id,
              sourceHandle: `s-${ii}`,
              targetHandle: layer.kind === "output" ? undefined : `t-${oi}`,
              type: "cnnWeight",
              // Paint above node chrome so edges aren't washed out by white fills.
              zIndex: 3,
              data: { weightMag: weightMagnitude(w), weight: w, active: true },
              style: {
                stroke: style.stroke,
                strokeWidth: style.strokeWidth,
                strokeOpacity: style.strokeOpacity,
              },
            });
          }
        }
      }
      // No hop edges between spatial layers (input/conv/pool) — layout alone is enough.
    }
    x += width + CNN_COL_GAP;
  });

  return { nodes, edges };
}
