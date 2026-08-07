/** Diverging palette: violet (neg) → orchid (0) → magenta (pos). */

export const PALETTE_LOW = { r: 58, g: 48, b: 168 }; // #3a30a8 — dark saturated violet
export const PALETTE_MID = { r: 168, g: 85, b: 196 }; // #a855c4 — vivid orchid
export const PALETTE_HIGH = { r: 192, g: 38, b: 112 }; // #c02670 — saturated magenta

export const CLASS_0_HEX = "#3a30a8";
export const CLASS_1_HEX = "#c02670";

const NUM_SHADES = 30;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
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

/** Maps a value in [-1, 1] to an RGB color (violet → light → magenta). */
export function valueToRgb(value: number): { r: number; g: number; b: number } {
  const t = (clamp(value, -1, 1) + 1) / 2;
  if (t <= 0.5) return interpolateColor(PALETTE_LOW, PALETTE_MID, t * 2);
  return interpolateColor(PALETTE_MID, PALETTE_HIGH, (t - 0.5) * 2);
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
