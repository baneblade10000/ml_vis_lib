import type { Edge, Node } from "@xyflow/react";
import { MarkerType } from "@xyflow/react";
import {
  weightColor,
  type CnnLayerView,
  type CnnMode,
  type FeatureMapSnapshot,
  type LayerKind,
  type LayerShape,
} from "@ml-vis/core";
import type { CnnMessages } from "./messages";

/** Minimal layer surface needed for localized titles. */
type LabelableLayer = { kind: LayerKind; label: () => string };

/** Pipeline view for React Flow (from train-worker snapshot or live engine). */
export type CnnPipelineView = {
  mode: CnnMode;
  layers: CnnLayerView[];
};

/** Pixel geometry for the CNN flow graph. */
export const CNN_COL_SPACING = 200;
export const CNN_ORIGIN_X = 80;
export const CNN_NODE_WIDTH = 150;

/** Cap displayed units so Flatten/Dense don't dominate the canvas. */
export const CNN_MAX_VIS_UNITS = 48;
export const CNN_MAX_STACK_HEIGHT = 140;

const MAP_PX = 44;
/** Pixel size of the kernel thumb drawn beside each conv channel. */
const KERNEL_PX = 36;
const MAP_GAP = 3;
const PAIR_GAP = 4;
const MAP_GRID_MAX_W = 140;
/** Wider wrap for conv nodes: kernel + activation side by side. */
const CONV_GRID_MAX_W = 280;
const NODE_CHROME = 38; // label + padding + gaps

/** Width of one conv channel cell: kernel thumb + gap + feature map. */
function convChannelCellW(mode: "2d" | "1d"): number {
  if (mode === "1d") return KERNEL_PX + PAIR_GAP + MAP_PX;
  return KERNEL_PX + PAIR_GAP + MAP_PX;
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

/** Circle stack size shared by Flatten / Dense weight columns. */
export function unitStackSize(
  length: number,
  columns = 1,
): { width: number; height: number; visCount: number; d: number; gap: number; gapX: number } {
  const n = Math.max(1, length);
  const visCount = Math.min(n, CNN_MAX_VIS_UNITS);
  const gap = 1.5;
  const gapX = 3;
  let d = Math.max(5.25, Math.min(10.5, (520 / visCount) * 1.5));
  let height = visCount * d + gap * Math.max(0, visCount - 1);
  if (height > CNN_MAX_STACK_HEIGHT) {
    d = Math.max(2.5, (CNN_MAX_STACK_HEIGHT - gap * Math.max(0, visCount - 1)) / visCount);
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
): { width: number; height: number } {
  if (kind === "flatten") {
    const n = shape?.kind === "1d" ? shape.length : 32;
    const stack = unitStackSize(n, 1);
    return { width: Math.max(48, stack.width + 28), height: NODE_CHROME + stack.height + 14 };
  }
  if (kind === "dense") {
    const units = shape?.kind === "1d" ? shape.length : 1;
    const stack = unitStackSize(Math.max(1, inputLength), Math.max(1, units));
    return {
      width: Math.max(64, stack.width + 24),
      height: NODE_CHROME + stack.height + 18,
    };
  }
  if (kind === "output") {
    return { width: CNN_NODE_WIDTH, height: 132 };
  }

  const channels = Math.max(1, Math.min(16, shape?.channels ?? 1));
  const isConv = kind === "conv2d" || kind === "conv1d";
  if (mode === "1d" || shape?.kind === "1d") {
    const rows = channels;
    const rowH = isConv ? Math.max(8, KERNEL_PX) : 8;
    const gridH = rows * rowH + Math.max(0, rows - 1) * MAP_GAP;
    const width = isConv
      ? Math.max(CNN_NODE_WIDTH, convChannelCellW("1d") + 28)
      : CNN_NODE_WIDTH;
    return { width, height: NODE_CHROME + gridH };
  }
  if (isConv) {
    const cellW = convChannelCellW("2d");
    const perRow = Math.max(1, Math.floor((CONV_GRID_MAX_W + MAP_GAP) / (cellW + MAP_GAP)));
    const gridRows = Math.ceil(channels / perRow);
    const gridH = gridRows * MAP_PX + Math.max(0, gridRows - 1) * MAP_GAP;
    const gridW = Math.min(channels, perRow) * cellW + Math.max(0, Math.min(channels, perRow) - 1) * MAP_GAP;
    return {
      width: Math.max(CNN_NODE_WIDTH, gridW + 28),
      height: NODE_CHROME + gridH,
    };
  }
  const perRow = Math.max(1, Math.floor((MAP_GRID_MAX_W + MAP_GAP) / (MAP_PX + MAP_GAP)));
  const gridRows = Math.ceil(channels / perRow);
  const gridH = gridRows * MAP_PX + Math.max(0, gridRows - 1) * MAP_GAP;
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

export type CnnEdgeData = { weightMag: number; active: boolean };

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
    return estimateNodeSize(layer.kind, shape, mode, inputLength);
  });
  const maxH = Math.max(1, ...sizes.map((s) => s.height));

  // Pack columns by actual node width so compact Flatten/Dense don't leave huge gaps.
  let x = CNN_ORIGIN_X;
  layers.forEach((layer, idx) => {
    const shape = layer.shape;
    const type = flowTypeFor(layer.kind);
    const isOutput = layer.kind === "output";
    const wMag = layer.weightMag;
    const { width, height } = sizes[idx]!;
    const labelable: LabelableLayer = { kind: layer.kind, label: () => layer.label };
    nodes.push({
      id: layer.id,
      type,
      position: {
        x,
        // Center every block on a shared horizontal axis.
        y: (maxH - height) / 2,
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
      const mag = wMag ?? 0;
      const stroke = weightColor(mag);
      const strokeW = 2 + Math.abs(mag) * 5.5;
      edges.push({
        id: `${prev.id}->${layer.id}`,
        source: prev.id,
        target: layer.id,
        type: "cnnWeight",
        data: { weightMag: mag, active: true },
        markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: stroke },
        style: { stroke, strokeWidth: strokeW, strokeOpacity: 0.45 + Math.abs(mag) * 0.55 },
      });
    }
    const nextW = sizes[idx + 1]?.width ?? width;
    x += Math.max(CNN_COL_SPACING, (width + nextW) / 2 + 56);
  });

  return { nodes, edges };
}
