/** Decision-boundary grid resolution for the output node. */
export const DENSITY = 400;
/** Coarse grid for input / hidden node thumbnails. */
export const NODE_BOUNDARY_DENSITY = 20;
/** During Play: sample every N cells on the output grid (N² fewer forward passes). */
export const PLAY_BOUNDARY_STRIDE = 8;
/** During Play: heatmap raster size before CSS upscale (matches stride on DENSITY=400: 400/8=50). */
export const PLAY_DISPLAY_DENSITY = 50;
/** 1D curve sample count for the output node. */
export const CURVE_DENSITY = 480;
/** 1D curve samples for input / hidden node thumbnails (must divide CURVE_DENSITY). */
export const NODE_CURVE_DENSITY = 160;
/**
 * During Play: subsample the output curve. 1D forwards are cheap, so keep this
 * at 1 — larger strides make the curve look staircase-y.
 */
export const PLAY_CURVE_STRIDE = 1;
export const X_DOMAIN: [number, number] = [-6, 6];
