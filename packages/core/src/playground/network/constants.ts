/** Named heatmap-resolution bundles (paused + play). */
export type HeatmapPresetId = "low" | "medium" | "high";

export type HeatmapPreset = {
  id: HeatmapPresetId;
  /** Paused output grid. Hidden density must divide this. */
  output: number;
  /** Paused input / hidden grid. */
  hidden: number;
  /** Play output samples (must divide {@link output}). */
  playOutput: number;
  /** Play hidden samples, block-filled into {@link hidden}. */
  playHidden: number;
};

export const HEATMAP_PRESETS: Record<HeatmapPresetId, HeatmapPreset> = {
  low: { id: "low", output: 100, hidden: 50, playOutput: 25, playHidden: 10 },
  medium: { id: "medium", output: 200, hidden: 100, playOutput: 50, playHidden: 20 },
  high: { id: "high", output: 400, hidden: 200, playOutput: 100, playHidden: 30 },
};

export const HEATMAP_PRESET_IDS: HeatmapPresetId[] = ["low", "medium", "high"];
export const DEFAULT_HEATMAP_PRESET: HeatmapPresetId = "medium";

export function heatmapPreset(id: HeatmapPresetId = DEFAULT_HEATMAP_PRESET): HeatmapPreset {
  return HEATMAP_PRESETS[id] ?? HEATMAP_PRESETS[DEFAULT_HEATMAP_PRESET];
}

export function playBoundaryStride(preset: HeatmapPreset): number {
  return Math.max(1, Math.round(preset.output / preset.playOutput));
}

const DEFAULT_PRESET = HEATMAP_PRESETS[DEFAULT_HEATMAP_PRESET];

/** Decision-boundary grid resolution for the output node (default preset). */
export const DENSITY = DEFAULT_PRESET.output;
/** Input / hidden node heatmaps when paused. Must divide {@link DENSITY}. */
export const NODE_BOUNDARY_DENSITY = DEFAULT_PRESET.hidden;
/** During Play: sample every N cells on the output grid (default: 200/4 → 50×50). */
export const PLAY_BOUNDARY_STRIDE = playBoundaryStride(DEFAULT_PRESET);
/** During Play: heatmap raster size before CSS upscale. */
export const PLAY_DISPLAY_DENSITY = DEFAULT_PRESET.playOutput;
/** During Play: hidden / input heatmap samples, block-filled into {@link NODE_BOUNDARY_DENSITY}. */
export const PLAY_NODE_BOUNDARY_DENSITY = DEFAULT_PRESET.playHidden;
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
