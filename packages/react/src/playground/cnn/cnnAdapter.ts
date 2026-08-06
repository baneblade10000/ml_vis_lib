import type { Edge, Node } from "@xyflow/react";
import { MarkerType } from "@xyflow/react";
import {
  CnnEngine,
  weightColor,
  type CnnMode,
  type FeatureMapSnapshot,
  type LayerKind,
  type LayerShape,
} from "@ml-vis/core";

/** Pixel geometry for the CNN flow graph. */
export const CNN_COL_SPACING = 240;
export const CNN_ORIGIN_X = 80;
export const CNN_NODE_WIDTH = 150;

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

/** Compute the aggregate weight magnitude for the edge feeding `layerId`. */
function edgeWeightMag(engine: CnnEngine, layerId: string): number | null {
  const layer = engine.layers.find((l) => l.id === layerId);
  if (!layer) return null;
  return layer.weightMagnitude();
}

export function cnnPipelineToFlow(
  engine: CnnEngine,
  options: {
    selectedNodeId: string | null;
    paintGeneration?: number;
    featureMaps: FeatureMapSnapshot[];
    loss?: number;
    probability?: number;
  },
): { nodes: Node<CnnNodeData>[]; edges: Edge<CnnEdgeData>[] } {
  const shapes = engine.pipelineShapes();
  const layers = engine.layers;
  const nodes: Node<CnnNodeData>[] = [];
  const edges: Edge<CnnEdgeData>[] = [];

  layers.forEach((layer, idx) => {
    const shape = shapes[idx];
    const type = flowTypeFor(layer.kind);
    const isOutput = layer.kind === "output";
    const wMag = edgeWeightMag(engine, layer.id);
    nodes.push({
      id: layer.id,
      type,
      position: { x: CNN_ORIGIN_X + idx * CNN_COL_SPACING, y: 0 },
      width: CNN_NODE_WIDTH,
      height: isOutput ? 132 : 150,
      data: {
        layerId: layer.id,
        kind: layer.kind,
        label: layer.label(),
        mode: engine.config.mode,
        channels: channelCount(shape),
        rows: shape?.kind === "2d" ? shape.rows : undefined,
        cols: shape?.kind === "2d" ? shape.cols : undefined,
        length: shape?.kind === "1d" ? shape.length : undefined,
        params: layer.paramCount(),
        weightMag: wMag,
        loss: isOutput ? options.loss : undefined,
        probability: isOutput ? options.probability : undefined,
        selected: options.selectedNodeId === layer.id,
        paintGeneration: options.paintGeneration,
      },
    });
    if (idx > 0) {
      const prev = layers[idx - 1];
      const mag = wMag ?? 0;
      const norm = mag; // already in [-1,1] via tanh
      const stroke = weightColor(norm);
      const width = 2 + Math.abs(mag) * 5.5;
      edges.push({
        id: `${prev.id}->${layer.id}`,
        source: prev.id,
        target: layer.id,
        type: "cnnWeight",
        data: { weightMag: mag, active: true },
        markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: stroke },
        style: { stroke, strokeWidth: width, strokeOpacity: 0.45 + Math.abs(mag) * 0.55 },
      });
    }
  });

  return { nodes, edges };
}

/** Drag kinds supported by the palette → inserted layer spec kind. */
export type CnnDragKind = "conv" | "pool" | "dense";

export function flowDragFromKind(kind: string, mode: CnnMode): CnnDragKind | null {
  if (kind === "conv" || kind === "pool" || kind === "dense") return kind;
  // Normalize "conv2d"/"conv1d" from a dropped source label.
  if (kind === "conv2d" || kind === "conv1d") return "conv";
  void mode;
  return null;
}

export const PALETTE_DRAG_TYPE = "application/reactflow-cnn";
