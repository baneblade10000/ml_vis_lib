/**
 * Numeric color scale — the one place that defines how a number becomes a color.
 *
 * A signed value in [-1, 1] maps onto a two-segment gradient through three
 * tokens: negative values blend from the negative token toward white, positive
 * ones from the positive token toward white, and 0 is exactly white. Schemes
 * with a meaning of their own (class labels, image pixels, series identity) are
 * not numeric and define their colors elsewhere.
 */

export interface Rgb {
  r: number;
  g: number;
  b: number;
}

/** Negative numbers: #3d41b9 indigo. */
export const NUMERIC_NEGATIVE: Rgb = { r: 61, g: 65, b: 185 };
/** Zero: white, so a neutral value disappears into a white surface. */
export const NUMERIC_ZERO: Rgb = { r: 255, g: 255, b: 255 };
/** Positive numbers: #1a90d9 cyan-blue. */
export const NUMERIC_POSITIVE: Rgb = { r: 26, g: 144, b: 217 };

function toHex({ r, g, b }: Rgb): string {
  return [r, g, b].map((c) => c.toString(16).padStart(2, "0")).join("");
}

const NEGATIVE_HEX = `#${toHex(NUMERIC_NEGATIVE)}`;
const POSITIVE_HEX = `#${toHex(NUMERIC_POSITIVE)}`;

/** Class labels sit on the scale endpoints so data points match their regions. */
export const CLASS_0_HEX = NEGATIVE_HEX;
export const CLASS_1_HEX = POSITIVE_HEX;

const NUM_SHADES = 30;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Pull mid-range values toward 0 so ±0.5 stay closer to the zero color instead
 * of snapping toward the extremes.
 */
const VALUE_TO_COLOR_GAMMA = 1.6;

function shapeForColor(value: number, gamma: number): number {
  const v = clamp(value, -1, 1);
  if (v === 0) return 0;
  return Math.sign(v) * Math.pow(Math.abs(v), gamma);
}

function interpolateColor(from: Rgb, to: Rgb, t: number): Rgb {
  return {
    r: Math.round(lerp(from.r, to.r, t)),
    g: Math.round(lerp(from.g, to.g, t)),
    b: Math.round(lerp(from.b, to.b, t)),
  };
}

/**
 * Maps a value in [-1, 1] onto the scale. `gamma` shapes how fast a value
 * leaves white: the default keeps mid-range values close to 0; `1` is a plain
 * linear blend toward the endpoint.
 */
export function valueToRgb(value: number, gamma = VALUE_TO_COLOR_GAMMA): Rgb {
  const shaped = shapeForColor(value, gamma);
  const endpoint = shaped < 0 ? NUMERIC_NEGATIVE : NUMERIC_POSITIVE;
  return interpolateColor(NUMERIC_ZERO, endpoint, Math.abs(shaped));
}

/** Maps a probability in [0, 1] to RGBA components (class 0 → class 1). */
export function probabilityToRgba(probability: number, alpha = 255): [number, number, number, number] {
  const { r, g, b } = valueToRgb(clamp(probability, 0, 1) * 2 - 1);
  return [r, g, b, alpha];
}

export function rgbToCss({ r, g, b }: Rgb): string {
  return `rgb(${r}, ${g}, ${b})`;
}

/** CSS color for a weight value in [-1, 1]. */
export function weightColor(value: number, gamma = VALUE_TO_COLOR_GAMMA): string {
  return rgbToCss(valueToRgb(value, gamma));
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

export function probabilityColorRgb(probability: number): Rgb {
  const [r, g, b] = probabilityToRgba(probability, 255);
  return { r, g, b };
}

/**
 * CSS gradient sampled from {@link valueToRgb}, for legends. The mapping is
 * shaped, so a three-stop `linear-gradient` would not match the canvases.
 */
export function numericScaleGradient(stops = 9): string {
  const samples = Array.from({ length: stops }, (_, i) => {
    const value = (2 * i) / (stops - 1) - 1;
    return `${rgbToCss(valueToRgb(value))} ${(100 * i) / (stops - 1)}%`;
  });
  return `linear-gradient(to right, ${samples.join(", ")})`;
}

const INK_ON_DARK = "#ffffff";
const INK_DARK_RGB: Rgb = { r: 15, g: 23, b: 42 };
const INK_ON_LIGHT = `#${toHex(INK_DARK_RGB)}`;

export function relativeLuminance({ r, g, b }: Rgb): number {
  const channel = (c: number) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

/** Whether dark ink has the higher WCAG contrast on this color. */
export function isPale(rgb: Rgb): boolean {
  const luminance = relativeLuminance(rgb);
  const darkContrast = (luminance + 0.05) / (relativeLuminance(INK_DARK_RGB) + 0.05);
  const whiteContrast = 1.05 / (luminance + 0.05);
  return darkContrast >= whiteContrast;
}

/**
 * Text or outline color that stays legible on the scale color at `value`:
 * values near 0 are white on the scale and take dark ink.
 */
export function scaleInk(value: number, gamma = VALUE_TO_COLOR_GAMMA): string {
  return isPale(valueToRgb(value, gamma)) ? INK_ON_LIGHT : INK_ON_DARK;
}

/**
 * Text color for a value written on a white surface, where the scale color may
 * be white too: the saturated token of the value's side, so the text matches
 * the scale and stays readable. 0 has no side and takes dark ink.
 */
export function scaleText(value: number): string {
  if (value === 0 || Number.isNaN(value)) return INK_ON_LIGHT;
  return value < 0 ? NEGATIVE_HEX : POSITIVE_HEX;
}
