/**
 * `@ml-vis/core` — root barrel.
 *
 * Aggregates only the framework-agnostic LEAF modules (layout, signal math,
 * canvas renderers, i18n, math utils). The ML engines live behind their own
 * subpaths so their same-named symbols never collide:
 *   - `DATASETS` / `DatasetId` exist in both the legacy MLP and network engines;
 *   - `Signal` exists in both the signal math and the CNN tensor module;
 *   - `PlaygroundOptimizerId` exists in both the CNN and network engines.
 * Import engines explicitly: `@ml-vis/core/network`, `/cnn`, `/autograd`,
 * `/mlp`, and the worker transport via `@ml-vis/core/workers`.
 */

export * from "./section";
export * from "./signal";
export * from "./charts";
export * from "./i18n";

export { downsample, extent } from "./utils/math";
