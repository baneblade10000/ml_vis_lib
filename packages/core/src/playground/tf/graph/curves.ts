import { constructInput, INPUTS } from "../inputs";
import { CURVE_DENSITY, NODE_CURVE_DENSITY, X_DOMAIN } from "../constants";
import { forwardPropGraph, forEachGraphNode, type ComputationalGraph } from "./computational-graph";

function scaleLinear(domain: [number, number], range: [number, number], value: number): number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

function nodeCurveDensity(graph: ComputationalGraph, nodeId: string): number {
  return nodeId === graph.outputId ? CURVE_DENSITY : NODE_CURVE_DENSITY;
}

/** Allocate 1D curve arrays (full res for output, coarse for other nodes). */
export function initGraphCurveStore(
  graph: ComputationalGraph,
  includeInputFeatures = true,
): Record<string, number[]> {
  const curves: Record<string, number[]> = {};
  forEachGraphNode(graph, true, (node) => {
    curves[node.id] = new Array<number>(nodeCurveDensity(graph, node.id)).fill(0);
  });
  if (includeInputFeatures) {
    for (const nodeId of Object.keys(INPUTS)) {
      curves[nodeId] = new Array<number>(NODE_CURVE_DENSITY).fill(0);
    }
  }
  return curves;
}

/**
 * Sample node outputs along x (y fixed at 0) into the curve store.
 * Output uses full density; other nodes are filled on a coarser grid.
 */
export function updateGraphCurves(
  graph: ComputationalGraph,
  activeInputs: Record<string, boolean>,
  curves: Record<string, number[]>,
  options?: {
    density?: number;
    xDomain?: [number, number];
    refreshInputFeatures?: boolean;
  },
): void {
  const density = options?.density ?? CURVE_DENSITY;
  const xDomain = options?.xDomain ?? X_DOMAIN;
  const refreshInputFeatures = options?.refreshInputFeatures ?? false;
  const miniFactor = density / NODE_CURVE_DENSITY;

  const xScale = (i: number) => scaleLinear([0, density - 1], xDomain, i);

  if (refreshInputFeatures) {
    for (let i = 0; i < NODE_CURVE_DENSITY; i++) {
      const x = scaleLinear([0, NODE_CURVE_DENSITY - 1], xDomain, i);
      for (const nodeId of Object.keys(INPUTS)) {
        if (curves[nodeId] && activeInputs[nodeId]) {
          curves[nodeId][i] = INPUTS[nodeId].f(x, 0);
        }
      }
    }
  }

  for (let i = 0; i < density; i++) {
    const x = xScale(i);
    const input = constructInput(x, 0, activeInputs);
    forwardPropGraph(graph, input, false);
    forEachGraphNode(graph, true, (node) => {
      const store = curves[node.id];
      if (!store) return;
      if (node.id === graph.outputId) {
        store[i] = node.output;
      } else if (i % miniFactor === 0) {
        store[i / miniFactor] = node.output;
      }
    });
  }
}

/** Fast hidden-node curve refresh at coarse density (for Play animation). */
export function updateGraphHiddenCurves(
  graph: ComputationalGraph,
  activeInputs: Record<string, boolean>,
  curves: Record<string, number[]>,
  outputId: string,
  xDomain: [number, number] = X_DOMAIN,
): void {
  const density = NODE_CURVE_DENSITY;
  const xScale = (i: number) => scaleLinear([0, density - 1], xDomain, i);

  for (let i = 0; i < density; i++) {
    const input = constructInput(xScale(i), 0, activeInputs);
    forwardPropGraph(graph, input, false);
    forEachGraphNode(graph, true, (node) => {
      if (node.id === outputId) return;
      const store = curves[node.id];
      if (store) store[i] = node.output;
    });
  }
}

/** Fast output-only curve refresh (for Play animation). Fills stride blocks. */
export function updateGraphOutputCurve(
  graph: ComputationalGraph,
  activeInputs: Record<string, boolean>,
  curves: Record<string, number[]>,
  outputId: string,
  stride = 1,
  xDomain: [number, number] = X_DOMAIN,
): void {
  const series = curves[outputId];
  if (!series?.length) return;
  const density = series.length;
  const step = Math.max(1, stride);
  const xScale = (i: number) => scaleLinear([0, density - 1], xDomain, i);

  for (let i = 0; i < density; i += step) {
    const input = constructInput(xScale(i), 0, activeInputs);
    forwardPropGraph(graph, input, false);
    const value = graph.getOutputNode().output;
    if (step === 1) {
      series[i] = value;
      continue;
    }
    for (let di = 0; di < step && i + di < density; di++) {
      series[i + di] = value;
    }
  }
}
