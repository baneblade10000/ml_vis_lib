/**
 * Canvas 2D renderers (charts). Public surface for the `@ml-vis/core/charts`
 * subpath. These are framework-agnostic drawing helpers used by the React
 * playgrounds and reusable directly.
 */

export { SignalPlot } from "./signal-plot";
export type { SignalMarker, SignalPlotSeries, SignalPlotPayload } from "./signal-plot";

export { DecisionBoundaryPlot } from "./decision-boundary-plot";
export type { DecisionBoundaryPayload } from "./decision-boundary-plot";

export { reduceMatrix, renderValueMatrix } from "./mini-heatmap";
export type {
  RenderValueMatrixOptions,
  ValueMatrixLayout,
  ValueMatrixPalette,
} from "./mini-heatmap";

export {
  curveStrokeFromValues,
  inferYDomain,
  renderCurve,
  renderCurvePoints,
  renderTargetCurve,
} from "./mini-curve";
