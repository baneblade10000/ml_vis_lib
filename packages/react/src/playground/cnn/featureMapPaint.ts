import {
  paintAllBoundaries,
  paintAllBoundariesAfterCommit,
  paintBoundaryNode,
  registerBoundaryPainter,
} from "../network/boundaryPaint";

/**
 * The convolutional playground reuses the network playground's imperative
 * painter registry (a module-level `Map<nodeId, Set<Painter>>`) rather than
 * duplicating it. These thin re-exports keep call sites within the `cnn/`
 * folder self-documenting while sharing the exact same plumbing.
 */
export { registerBoundaryPainter };

/** Paint every registered feature-map canvas now. */
export function paintAllFeatureMaps(): void {
  paintAllBoundaries();
}

/** Paint now, after a microtask, and after a frame — for React Flow DOM churn. */
export function paintAllFeatureMapsAfterCommit(): void {
  paintAllBoundariesAfterCommit();
}

/** Paint only the canvases registered for one node. */
export function paintFeatureMapNode(nodeId: string): void {
  paintBoundaryNode(nodeId);
}
