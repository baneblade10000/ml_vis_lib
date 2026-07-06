/** Diverging palette: purple (negative / class 0) → neutral → crimson (positive / class 1). */

export const PALETTE_LOW = { r: 79, g: 70, b: 229 }; // #4f46e5 purple
export const PALETTE_MID = { r: 236, g: 237, b: 240 }; // #ecedf0
export const PALETTE_HIGH = { r: 220, g: 38, b: 58 }; // #dc263a crimson

export const CLASS_0_HEX = "#4f46e5";
export const CLASS_1_HEX = "#dc263a";

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

/** Maps a value in [-1, 1] to an RGB color. */
export function valueToRgb(value: number): { r: number; g: number; b: number } {
  const v = clamp(value, -1, 1);
  const normalized = (v + 1) / 2;
  if (normalized <= 0.5) {
    const t = normalized / 0.5;
    return interpolateColor(PALETTE_LOW, PALETTE_MID, t);
  }
  const t = (normalized - 0.5) / 0.5;
  return interpolateColor(PALETTE_MID, PALETTE_HIGH, t);
}

/** Maps a probability in [0, 1] to RGBA components. */
export function probabilityToRgba(probability: number, alpha = 160): [number, number, number, number] {
  const p = clamp(probability, 0, 1);
  let r: number;
  let g: number;
  let b: number;
  if (p <= 0.5) {
    const t = p / 0.5;
    r = PALETTE_LOW.r + (PALETTE_MID.r - PALETTE_LOW.r) * t;
    g = PALETTE_LOW.g + (PALETTE_MID.g - PALETTE_LOW.g) * t;
    b = PALETTE_LOW.b + (PALETTE_MID.b - PALETTE_LOW.b) * t;
  } else {
    const t = (p - 0.5) / 0.5;
    r = PALETTE_MID.r + (PALETTE_HIGH.r - PALETTE_MID.r) * t;
    g = PALETTE_MID.g + (PALETTE_HIGH.g - PALETTE_MID.g) * t;
    b = PALETTE_MID.b + (PALETTE_HIGH.b - PALETTE_MID.b) * t;
  }
  return [Math.round(r), Math.round(g), Math.round(b), alpha];
}

/** CSS color string for a weight value in [-1, 1]. */
export function weightColor(value: number): string {
  const { r, g, b } = valueToRgb(value);
  return `rgb(${r}, ${g}, ${b})`;
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
