/** Diverging palette: deep blue (neg) → mid blue (0) → sky cyan (pos). */

export const PALETTE_LOW = { r: 8, g: 50, b: 117 }; // #083275
export const PALETTE_MID = { r: 51, g: 125, b: 186 }; // #337dba — blend of endpoints
export const PALETTE_HIGH = { r: 94, g: 201, b: 255 }; // #5ec9ff

export const CLASS_0_HEX = "#083275";
export const CLASS_1_HEX = "#5ec9ff";

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

/** Maps a value in [-1, 1] to an RGB color (deep blue → mid → sky cyan). */
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
