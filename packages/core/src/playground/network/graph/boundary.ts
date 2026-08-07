import { constructInput, INPUTS } from "../inputs";
import { DENSITY, NODE_BOUNDARY_DENSITY, X_DOMAIN } from "../constants";
import { forwardPropGraph, forEachGraphNode, type ComputationalGraph } from "./computational-graph";

function scaleLinear(domain: [number, number], range: [number, number], value: number): number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

function nodeBoundaryDensity(graph: ComputationalGraph, nodeId: string): number {
  return nodeId === graph.outputId ? DENSITY : NODE_BOUNDARY_DENSITY;
}

/** Allocate boundary matrices (full res for output, coarse for other nodes). */
export function initGraphBoundaryStore(
  graph: ComputationalGraph,
  includeInputFeatures = true,
): Record<string, number[][]> {
  const boundary: Record<string, number[][]> = {};
  forEachGraphNode(graph, true, (node) => {
    const density = nodeBoundaryDensity(graph, node.id);
    boundary[node.id] = Array.from({ length: density }, () => new Array<number>(density));
  });
  if (includeInputFeatures) {
    for (const nodeId of Object.keys(INPUTS)) {
      boundary[nodeId] = Array.from({ length: NODE_BOUNDARY_DENSITY }, () =>
        new Array<number>(NODE_BOUNDARY_DENSITY),
      );
    }
  }
  return boundary;
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
    xDomain?: [number, number];
    /** Re-sample static input-feature surfaces (only needed when inputs change). */
    refreshInputFeatures?: boolean;
  },
): void {
  const density = options?.density ?? DENSITY;
  const xDomain = options?.xDomain ?? X_DOMAIN;
  const refreshInputFeatures = options?.refreshInputFeatures ?? false;
  const miniFactor = density / NODE_BOUNDARY_DENSITY;

  const xScale = (i: number) => scaleLinear([0, density - 1], xDomain, i);
  const yScale = (j: number) => scaleLinear([density - 1, 0], xDomain, j);

  if (refreshInputFeatures) {
    const fScale = (i: number) =>
      scaleLinear([0, NODE_BOUNDARY_DENSITY - 1], xDomain, i);
    for (let i = 0; i < NODE_BOUNDARY_DENSITY; i++) {
      for (let j = 0; j < NODE_BOUNDARY_DENSITY; j++) {
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
): Record<string, number[][]> {
  const boundary = initGraphBoundaryStore(graph, includeInputFeatures);
  updateGraphBoundaries(graph, activeInputs, boundary, {
    density,
    xDomain,
    refreshInputFeatures: includeInputFeatures,
  });
  return boundary;
}

/**
 * Fast hidden-node boundary refresh at coarse density (for animation).
 * Only 10×10 forward passes, so it can run every paint frame during Play.
 */
export function updateGraphHiddenBoundaries(
  graph: ComputationalGraph,
  activeInputs: Record<string, boolean>,
  boundary: Record<string, number[][]>,
  outputId: string,
  xDomain: [number, number] = X_DOMAIN,
): void {
  const density = NODE_BOUNDARY_DENSITY;
  const xScale = (i: number) => scaleLinear([0, density - 1], xDomain, i);
  const yScale = (j: number) => scaleLinear([density - 1, 0], xDomain, j);

  for (let i = 0; i < density; i++) {
    for (let j = 0; j < density; j++) {
      const input = constructInput(xScale(i), yScale(j), activeInputs);
      forwardPropGraph(graph, input, false);
      forEachGraphNode(graph, true, (node) => {
        if (node.id === outputId) return;
        const store = boundary[node.id];
        if (store) store[i]![j] = node.output;
      });
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
