/** Diverging palette: #3d41b9 indigo (neg) → #6999fa light-blue (0) → #1a90d9 cyan-blue (pos). */

export const PALETTE_LOW = { r: 61, g: 65, b: 185 }; // #3d41b9
export const PALETTE_MID = { r: 105, g: 153, b: 250 }; // #6999fa — midpoint
/** White midpoint for signed learnable params (CNN kernels / weights / biases). */
export const PALETTE_ZERO_WHITE = { r: 255, g: 255, b: 255 };
export const PALETTE_HIGH = { r: 26, g: 144, b: 217 }; // #1a90d9

export const CLASS_0_HEX = "#3d41b9";
export const CLASS_1_HEX = "#1a90d9";

const NUM_SHADES = 30;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Pull mid-range values toward 0 so ±0.5 stay closer to the midpoint color
 * instead of snapping toward the extremes.
 */
const VALUE_TO_COLOR_GAMMA = 1.6;

function shapeForColor(value: number): number {
  const v = clamp(value, -1, 1);
  if (v === 0) return 0;
  return Math.sign(v) * Math.pow(Math.abs(v), VALUE_TO_COLOR_GAMMA);
}

function interpolateColor(
  from: { r: number; g: number; b: number },
  to: { r: number; g: number; b: number },
  t: number,
): { r: number; g: number; b: number } {
  return {
    r: Math.round(lerp(from.r, to.r, t)),
    g: Math.round(lerp(from.g, to.g, t)),
    b: Math.round(lerp(from.b, to.b, t)),
  };
}

function mapValueToRgb(
  value: number,
  mid: { r: number; g: number; b: number },
): { r: number; g: number; b: number } {
  const t = (shapeForColor(value) + 1) / 2;
  if (t <= 0.5) return interpolateColor(PALETTE_LOW, mid, t * 2);
  return interpolateColor(mid, PALETTE_HIGH, (t - 0.5) * 2);
}

/** Maps a value in [-1, 1] to an RGB color (deep blue → mid → sky cyan). */
export function valueToRgb(value: number): { r: number; g: number; b: number } {
  return mapValueToRgb(value, PALETTE_MID);
}

/** Same as {@link valueToRgb}, but 0 is white (CNN learnable parameters). */
export function valueToRgbZeroWhite(value: number): { r: number; g: number; b: number } {
  return mapValueToRgb(value, PALETTE_ZERO_WHITE);
}

/** Maps a probability in [0, 1] to RGBA components (class 0 → class 1). */
export function probabilityToRgba(probability: number, alpha = 255): [number, number, number, number] {
  const { r, g, b } = valueToRgb(clamp(probability, 0, 1) * 2 - 1);
  return [r, g, b, alpha];
}

/** CSS color string for a weight value in [-1, 1]. */
export function weightColor(value: number): string {
  const { r, g, b } = valueToRgb(value);
  return `rgb(${r}, ${g}, ${b})`;
}

/** CSS color for a signed param in [-1, 1] with white at 0. */
export function weightColorZeroWhite(value: number): string {
  const { r, g, b } = valueToRgbZeroWhite(value);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Smoothly normalized weight magnitude in [0, 1]. `tanh` keeps small weights
 * legible while saturating gracefully for large ones, so that color, stroke
 * width and opacity all derive from the *same* scale instead of clamping at
 * different thresholds.
 */
export function weightMagnitude(weight: number): number {
  return Math.tanh(Math.abs(weight));
}

/** Sign-aware normalized weight in (-1, 1), for diverging color lookup. */
export function weightValueNormalized(weight: number): number {
  return Math.sign(weight) * weightMagnitude(weight);
}

/** Quantized heatmap palette (30 shades). */
export function weightColorQuantized(value: number): string {
  const v = clamp(value, -1, 1);
  const normalized = (v + 1) / 2;
  const step = Math.round(normalized * NUM_SHADES) / NUM_SHADES;
  return weightColor(step * 2 - 1);
}

export function mixProbabilityColor(probability: number, alpha = 0.63): string {
  const [r, g, b, a] = probabilityToRgba(probability, Math.round(alpha * 255));
  return `rgba(${r}, ${g}, ${b}, ${(a / 255).toFixed(3)})`;
}

export function probabilityColorRgb(probability: number): { r: number; g: number; b: number } {
  const [r, g, b] = probabilityToRgba(probability, 255);
  return { r, g, b };
}
