import type { GraphNode } from "./runtime";
import type { ComputationalGraph } from "./computational-graph";
import type { GraphPosition } from "./types";

/** Minimum left-edge distance between MLP columns (input → hidden → … → output). */
export const MLP_COL_SPACING = 180;
/** Vertical gap between neurons within one layer. */
export const MLP_ROW_SPACING = 88;
export const MLP_ORIGIN_X = 72;
export const MLP_ORIGIN_Y = 80;
/** Rendered size of regular nodes (px, must match react NODE_WIDTH). */
export const MLP_NODE_SIZE = 52;
/** Rendered size of the output node (px, must match react OUTPUT_NODE_WIDTH). */
export const MLP_OUTPUT_NODE_SIZE = 218;
/** Y offset that vertically centers the large output node on the row axis. */
const OUTPUT_Y_OFFSET = (MLP_OUTPUT_NODE_SIZE - MLP_NODE_SIZE) / 2;

/**
 * Left-edge distance between two columns. Grows with the fully-connected pair
 * size so dense edge bundles have room instead of smearing into a solid mass.
 * A 16×1 fan-in (hidden → output) stays at the default; 16×16 opens up.
 */
export function mlpLayerGap(srcCount: number, destCount: number): number {
  const pair = Math.sqrt(Math.max(srcCount, 1) * Math.max(destCount, 1));
  const span = Math.max(0, pair - 1) * MLP_ROW_SPACING;
  return Math.max(MLP_COL_SPACING, Math.round(span * 0.42) + MLP_NODE_SIZE);
}

/** Column left-edges for a stack of layers with the given neuron counts. */
export function mlpColumnXsFromCounts(counts: number[], originX = MLP_ORIGIN_X): number[] {
  if (!counts.length) return [];
  const xs: number[] = [originX];
  for (let i = 0; i < counts.length - 1; i++) {
    xs.push(xs[i]! + mlpLayerGap(counts[i]!, counts[i + 1]!));
  }
  return xs;
}

export function mlpColumnX(columnIndex: number): number {
  return MLP_ORIGIN_X + columnIndex * MLP_COL_SPACING;
}

/** Evenly stack `count` nodes around `anchorY`. */
export function mlpStackYs(count: number, anchorY = MLP_ORIGIN_Y): number[] {
  if (count <= 0) return [];
  if (count === 1) return [anchorY];
  const span = (count - 1) * MLP_ROW_SPACING;
  const y0 = anchorY - span / 2;
  return Array.from({ length: count }, (_, i) => y0 + i * MLP_ROW_SPACING);
}

export function mlpOutputAnchorY(layers: GraphNode[][]): number {
  const counts = [
    layers[0]?.length ?? 0,
    ...layers.slice(1, -1).map((layer) => layer.length),
  ];
  const maxCount = Math.max(...counts, 1);
  const ys = mlpStackYs(maxCount);
  return ys[Math.floor((ys.length - 1) / 2)] ?? MLP_ORIGIN_Y;
}

/** Position nodes in a single column; other columns are untouched. */
export function layoutMlpColumn(
  graph: ComputationalGraph,
  columnIndex: number,
  nodeIds: string[],
  anchorY?: number,
  x = mlpColumnX(columnIndex),
): void {
  const ys = mlpStackYs(nodeIds.length, anchorY);
  nodeIds.forEach((id, i) => {
    graph.setPosition(id, { x, y: ys[i] });
  });
}

export function layoutMlpFromLayers(
  graph: ComputationalGraph,
  layers: GraphNode[][],
  inputIds: string[],
): void {
  const anchorY = mlpOutputAnchorY(layers);
  const counts = layers.map((layer, i) => (i === 0 ? inputIds.length : layer.length));
  const colXs = mlpColumnXsFromCounts(counts);

  layoutMlpColumn(graph, 0, inputIds, anchorY, colXs[0]);

  for (let layerIdx = 1; layerIdx < layers.length; layerIdx++) {
    const layer = layers[layerIdx];
    const x = colXs[layerIdx] ?? mlpColumnX(layerIdx);
    const isOutput = layerIdx === layers.length - 1 && layer.length === 1;
    if (isOutput) {
      graph.setPosition(layer[0].id, { x, y: anchorY - OUTPUT_Y_OFFSET });
    } else {
      layoutMlpColumn(
        graph,
        layerIdx,
        layer.map((n) => n.id),
        anchorY,
        x,
      );
    }
  }
}

export function readNodePosition(graph: ComputationalGraph, nodeId: string): GraphPosition | undefined {
  return graph.positions.get(nodeId);
}

/**
 * Reset every node to the canonical column grid derived from graph topology.
 * Columns follow feed-forward depth (inputs → … → output); nodes in the same
 * column are stacked vertically around a shared anchor.
 */
export function normalizeGraphLayout(graph: ComputationalGraph, inputIds?: string[]): void {
  const ids = inputIds ?? [...graph.inputIds];
  layoutMlpFromLayers(graph, graph.toLayeredNetwork(), ids);
}
