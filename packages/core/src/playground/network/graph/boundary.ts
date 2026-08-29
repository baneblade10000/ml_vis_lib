import { constructInput, INPUTS } from "../inputs";
import { DENSITY, NODE_BOUNDARY_DENSITY, PLAY_NODE_BOUNDARY_DENSITY, X_DOMAIN } from "../constants";
import { forwardPropGraph, forEachGraphNode, type ComputationalGraph } from "./computational-graph";

export type BoundaryDensities = { output: number; hidden: number };

/** Live play raster for one node: `data[x * res + y]` over the input grid. */
export type BoundaryTile = { res: number; data: Float32Array };
export type BoundaryTileStore = Record<string, BoundaryTile>;

function scaleLinear(domain: [number, number], range: [number, number], value: number): number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

function resolveDensities(dens?: Partial<BoundaryDensities>): BoundaryDensities {
  return {
    output: dens?.output ?? DENSITY,
    hidden: dens?.hidden ?? NODE_BOUNDARY_DENSITY,
  };
}

function nodeBoundaryDensity(
  graph: ComputationalGraph,
  nodeId: string,
  dens: BoundaryDensities,
): number {
  return nodeId === graph.outputId ? dens.output : dens.hidden;
}

/** Allocate boundary matrices (full res for output, coarse for other nodes). */
export function initGraphBoundaryStore(
  graph: ComputationalGraph,
  includeInputFeatures = true,
  dens?: Partial<BoundaryDensities>,
): Record<string, number[][]> {
  const d = resolveDensities(dens);
  const boundary: Record<string, number[][]> = {};
  forEachGraphNode(graph, true, (node) => {
    const density = nodeBoundaryDensity(graph, node.id, d);
    boundary[node.id] = Array.from({ length: density }, () => new Array<number>(density));
  });
  if (includeInputFeatures) {
    for (const nodeId of Object.keys(INPUTS)) {
      boundary[nodeId] = Array.from({ length: d.hidden }, () => new Array<number>(d.hidden));
    }
  }
  return boundary;
}

export function updateGraphInputFeatures(
  activeInputs: Record<string, boolean>,
  boundary: Record<string, number[][]>,
  hiddenDensity = NODE_BOUNDARY_DENSITY,
  xDomain: [number, number] = X_DOMAIN,
): void {
  const fScale = (i: number) => scaleLinear([0, hiddenDensity - 1], xDomain, i);
  for (let i = 0; i < hiddenDensity; i++) {
    for (let j = 0; j < hiddenDensity; j++) {
      const x = fScale(i);
      const y = fScale(j);
      for (const nodeId of Object.keys(INPUTS)) {
        if (boundary[nodeId] && activeInputs[nodeId]) {
          boundary[nodeId][i]![j] = INPUTS[nodeId].f(x, y);
        }
      }
    }
  }
}

/**
 * Update boundary matrices in place with a single forward-prop sweep per grid cell
 * (same strategy as the original interactive playground).
 */
export function updateGraphBoundaries(
  graph: ComputationalGraph,
  activeInputs: Record<string, boolean>,
  boundary: Record<string, number[][]>,
  options?: {
    density?: number;
    hiddenDensity?: number;
    xDomain?: [number, number];
    /** Re-sample static input-feature surfaces (only needed when inputs change). */
    refreshInputFeatures?: boolean;
  },
): void {
  const density = options?.density ?? DENSITY;
  const hiddenDensity = options?.hiddenDensity ?? NODE_BOUNDARY_DENSITY;
  const xDomain = options?.xDomain ?? X_DOMAIN;
  const refreshInputFeatures = options?.refreshInputFeatures ?? false;
  const miniFactor = density / hiddenDensity;

  const xScale = (i: number) => scaleLinear([0, density - 1], xDomain, i);
  const yScale = (j: number) => scaleLinear([density - 1, 0], xDomain, j);

  if (refreshInputFeatures) {
    updateGraphInputFeatures(activeInputs, boundary, hiddenDensity, xDomain);
  }

  for (let i = 0; i < density; i++) {
    for (let j = 0; j < density; j++) {
      const x = xScale(i);
      const y = yScale(j);
      const input = constructInput(x, y, activeInputs);
      forwardPropGraph(graph, input, false);
      forEachGraphNode(graph, true, (node) => {
        const store = boundary[node.id];
        if (!store) return;
        if (node.id === graph.outputId) {
          store[i]![j] = node.output;
        } else if (i % miniFactor === 0 && j % miniFactor === 0) {
          store[i / miniFactor]![j / miniFactor] = node.output;
        }
      });
    }
  }
}

/** Build fresh boundary store (alloc + sample). */
export function computeGraphBoundaries(
  graph: ComputationalGraph,
  activeInputs: Record<string, boolean>,
  density = DENSITY,
  xDomain: [number, number] = X_DOMAIN,
  includeInputFeatures = true,
  hiddenDensity = NODE_BOUNDARY_DENSITY,
): Record<string, number[][]> {
  const boundary = initGraphBoundaryStore(graph, includeInputFeatures, {
    output: density,
    hidden: hiddenDensity,
  });
  updateGraphBoundaries(graph, activeInputs, boundary, {
    density,
    hiddenDensity,
    xDomain,
    refreshInputFeatures: includeInputFeatures,
  });
  return boundary;
}

/**
 * Fast hidden-node boundary refresh at {@link PLAY_NODE_BOUNDARY_DENSITY}.
 * Samples are block-filled into the paused-size store so Play stays cheap
 * without reallocating matrices.
 */
export function updateGraphHiddenBoundaries(
  graph: ComputationalGraph,
  activeInputs: Record<string, boolean>,
  boundary: Record<string, number[][]>,
  outputId: string,
  playDensity = PLAY_NODE_BOUNDARY_DENSITY,
  xDomain: [number, number] = X_DOMAIN,
): void {
  const density = playDensity;
  const xScale = (i: number) => scaleLinear([0, density - 1], xDomain, i);
  const yScale = (j: number) => scaleLinear([density - 1, 0], xDomain, j);

  for (let i = 0; i < density; i++) {
    for (let j = 0; j < density; j++) {
      const input = constructInput(xScale(i), yScale(j), activeInputs);
      forwardPropGraph(graph, input, false);
      forEachGraphNode(graph, true, (node) => {
        if (node.id === outputId) return;
        const store = boundary[node.id];
        if (store) fillPlayBlock(store, i, j, density, node.output);
      });
    }
  }
}

/** Nearest-neighbor upsample one Play sample into the paused-size store. */
function fillPlayBlock(
  store: number[][],
  i: number,
  j: number,
  playDensity: number,
  value: number,
): void {
  const n = store.length;
  if (!n) return;
  const i0 = Math.floor((i * n) / playDensity);
  const i1 = Math.max(i0 + 1, Math.floor(((i + 1) * n) / playDensity));
  const j0 = Math.floor((j * n) / playDensity);
  const j1 = Math.max(j0 + 1, Math.floor(((j + 1) * n) / playDensity));
  for (let ii = i0; ii < i1 && ii < n; ii++) {
    for (let jj = j0; jj < j1 && jj < n; jj++) {
      store[ii]![jj] = value;
    }
  }
}

/** Fast output-only boundary refresh (for animation). Fills stride×stride blocks per sample. */
export function updateGraphOutputBoundary(
  graph: ComputationalGraph,
  activeInputs: Record<string, boolean>,
  boundary: Record<string, number[][]>,
  outputId: string,
  stride = 1,
  xDomain: [number, number] = X_DOMAIN,
): void {
  const matrix = boundary[outputId];
  if (!matrix?.length) return;
  const density = matrix.length;
  const step = Math.max(1, stride);

  const xScale = (i: number) => scaleLinear([0, density - 1], xDomain, i);
  const yScale = (j: number) => scaleLinear([density - 1, 0], xDomain, j);

  for (let i = 0; i < density; i += step) {
    for (let j = 0; j < density; j += step) {
      const input = constructInput(xScale(i), yScale(j), activeInputs);
      forwardPropGraph(graph, input, false);
      const value = graph.getOutputNode().output;
      if (step === 1) {
        matrix[i]![j] = value;
        continue;
      }
      for (let di = 0; di < step && i + di < density; di++) {
        for (let dj = 0; dj < step && j + dj < density; dj++) {
          matrix[i + di]![j + dj] = value;
        }
      }
    }
  }
}

/**
 * Sample Play-density boundary tiles without touching the paused-size store.
 *
 * Same forward-prop sweeps as {@link updateGraphOutputBoundary} +
 * {@link updateGraphHiddenBoundaries}, but values land in small Float32Arrays
 * sized to the Play densities — ready for a zero-copy `postMessage` transfer
 * and a canvas upscale, instead of a block-filled full-res nested-matrix clone.
 * Input-feature nodes are skipped: their surfaces are static during training,
 * so the UI keeps the full-resolution matrices from the last full snapshot.
 */
export function sampleGraphBoundaryTiles(
  graph: ComputationalGraph,
  activeInputs: Record<string, boolean>,
  outputId: string,
  dens: { playOutput: number; playHidden: number },
  xDomain: [number, number] = X_DOMAIN,
): BoundaryTileStore {
  const tiles: BoundaryTileStore = {};
  // forEachGraphNode(ignoreInputs=true): input-feature surfaces are static
  // during training — the UI keeps their full-res matrices from full snapshots.
  forEachGraphNode(graph, true, (node) => {
    const res = node.id === outputId ? dens.playOutput : dens.playHidden;
    tiles[node.id] = { res, data: new Float32Array(res * res) };
  });
  const outputTile = tiles[outputId];
  if (outputTile) {
    const density = outputTile.res;
    const xScale = (i: number) => scaleLinear([0, density - 1], xDomain, i);
    const yScale = (j: number) => scaleLinear([density - 1, 0], xDomain, j);
    for (let i = 0; i < density; i++) {
      for (let j = 0; j < density; j++) {
        const input = constructInput(xScale(i), yScale(j), activeInputs);
        forwardPropGraph(graph, input, false);
        outputTile.data[i * density + j] = graph.getOutputNode().output;
      }
    }
  }
  const hidden = Object.entries(tiles).filter(([id]) => id !== outputId);
  if (hidden.length > 0) {
    const density = hidden[0]![1].res;
    const xScale = (i: number) => scaleLinear([0, density - 1], xDomain, i);
    const yScale = (j: number) => scaleLinear([density - 1, 0], xDomain, j);
    for (let i = 0; i < density; i++) {
      for (let j = 0; j < density; j++) {
        const input = constructInput(xScale(i), yScale(j), activeInputs);
        forwardPropGraph(graph, input, false);
        forEachGraphNode(graph, true, (node) => {
          if (node.id === outputId) return;
          const tile = tiles[node.id];
          if (tile) tile.data[i * density + j] = node.output;
        });
      }
    }
  }
  return tiles;
}
