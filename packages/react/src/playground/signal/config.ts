/**
 * Shared config, colour palette, and per-tab payload builders for the Signal lab.
 *
 * The math lives in `@ml-vis/core` (signal presets, convolve/correlate, FFT);
 * these pure functions turn a {@link SignalLabState} into a {@link SignalPlotPayload}
 * for whichever tab is active. Keeping them framework-agnostic makes them trivial
 * to reason about and reuse.
 */

import { buildSignal, convolve, convolveViaFft, correlate, dftMagnitude, type Signal as CoreSignal, type SignalParamId, type SignalPresetId } from "@ml-vis/core/signal";
import { type SignalPlotPayload } from "@ml-vis/core/charts";
import type { SignalTabId } from "./messages";

export type { SignalTabId, SignalPresetId, SignalParamId };

export interface SignalInput {
  preset: SignalPresetId;
  params: Partial<Record<SignalParamId, number>>;
}

export interface SignalLabState {
  f: SignalInput;
  g: SignalInput;
  /** Sliding-kernel position (sample index) for the convolution/correlation tabs. */
  position: number;
  /** Fourier tab: selected harmonic bin. */
  harmonic: number;
  correlationShowConvolution: boolean;
  /** Length of every generated signal. */
  length: number;
}

/** Palette — indigo/crimson accents match the app design tokens. */
export const COLOR = {
  f: "#4f46e5",
  g: "#dc263a",
  result: "#0f766e",
  direct: "#0f766e",
  fft: "#7c3aed",
  harmonic: "#f59e0b",
  convOverlay: "#0f766e",
  position: "rgba(15, 23, 42, 0.55)",
} as const;

export const DEFAULT_LENGTH = 64;

export const DEFAULT_STATE: SignalLabState = {
  f: { preset: "expDecay", params: { tau: 3 } },
  g: { preset: "gaussian", params: { sigma: 2.5 } },
  position: 0,
  harmonic: 1,
  correlationShowConvolution: false,
  length: DEFAULT_LENGTH,
};

export function buildInputs(state: SignalLabState): { f: CoreSignal; g: CoreSignal } {
  return {
    f: buildSignal(state.f.preset, state.f.params, state.length),
    g: buildSignal(state.g.preset, state.g.params, state.length),
  };
}

/**
 * Convolution tab: the kernel is drawn flipped and translated to the current
 * sliding position, plus the partial (accumulated) product so far and the full
 * result. The whole point is to *see* the flip that distinguishes convolution.
 */
export function buildConvolutionPayload(
  state: SignalLabState,
  f: CoreSignal,
  g: CoreSignal,
  labels: { f: string; g: string; result: string; position: string },
): SignalPlotPayload {
  const conv = convolve(f, g);
  const pos = clamp(state.position, 0, conv.length - 1);
  const series: SignalPlotPayload["series"] = [
    { values: padTo(f.values, conv.length), color: COLOR.f, label: labels.f, fill: true },
    { values: flippedKernel(g, pos, conv.length), color: COLOR.g, label: labels.g },
    { values: conv.values, color: COLOR.result, label: labels.result, fill: true },
  ];
  return {
    series,
    markers: [{ x: pos, color: COLOR.position, label: labels.position }],
    subtitle: "y[n] = Σ_m f[m] · g[n − m]",
  };
}

/**
 * Cross-correlation tab: identical inputs, but the kernel is NOT flipped. A
 * toggle overlays the convolution result so the single-flip difference is
 * unmistakable.
 */
export function buildCorrelationPayload(
  state: SignalLabState,
  f: CoreSignal,
  g: CoreSignal,
  labels: { f: string; g: string; correlation: string; convolution: string; position: string },
): SignalPlotPayload {
  const corr = correlate(f, g);
  const conv = convolve(f, g);
  const pos = clamp(state.position, 0, corr.length - 1);
  const series: SignalPlotPayload["series"] = [
    { values: padTo(f.values, corr.length), color: COLOR.f, label: labels.f, fill: true },
    { values: translatedKernel(g, pos, corr.length), color: COLOR.g, label: labels.g },
    { values: corr.values, color: COLOR.result, label: labels.correlation, fill: true },
  ];
  if (state.correlationShowConvolution) {
    series.push({
      values: conv.values,
      color: COLOR.convOverlay,
      label: labels.convolution,
      dashed: true,
    });
  }
  return {
    series,
    markers: [{ x: pos, color: COLOR.position, label: labels.position }],
    subtitle: "y[n] = Σ_m f[m] · g[m − n]   (kernel not flipped)",
  };
}

/**
 * Fourier tab: the signal plus its magnitude spectrum as stems, and the selected
 * harmonic reconstructed and overlaid so users see which sinusoid that bin is.
 */
export function buildFourierPayload(
  state: SignalLabState,
  f: CoreSignal,
  labels: { f: string; spectrum: string; harmonic: string },
): SignalPlotPayload {
  const spectrum = dftMagnitude(f.values);
  const k = clamp(state.harmonic, 0, f.length - 1);
  const cos = dftCosine(f.values, k);
  const sin = -dftSine(f.values, k);
  const phase = Math.atan2(sin, cos);
  // Unit-amplitude sinusoid for bin k, phase-aligned with the signal's component.
  const harmonic = new Float64Array(f.length);
  for (let i = 0; i < f.length; i++) {
    harmonic[i] = Math.cos((2 * Math.PI * k * i) / f.length + phase);
  }
  return {
    series: [
      { values: harmonic, color: COLOR.harmonic, label: labels.harmonic },
      { values: f.values, color: COLOR.f, label: labels.f, fill: true },
      {
        values: spectrum.magnitudes,
        color: COLOR.fft,
        label: labels.spectrum,
        style: "stem",
      },
    ],
    subtitle: `f[i] = (1/N) Σ_k |F[k]|·cos(2πki/N + φ_k)   —   k = ${k}`,
    caption: labels.spectrum,
  };
}

/** Convolution theorem tab: direct convolution vs ifft(fft(f)·fft(g)). */
export function buildTheoremPayload(
  f: CoreSignal,
  g: CoreSignal,
  labels: {
    f: string;
    g: string;
    direct: string;
    fft: string;
    recipe: string;
    deviation: string;
  },
): SignalPlotPayload {
  const direct = convolve(f, g);
  const viaFft = convolveViaFft(f.values, g.values);
  let worst = 0;
  for (let i = 0; i < direct.length; i++) {
    worst = Math.max(worst, Math.abs(direct.values[i] - viaFft.result[i]));
  }
  return {
    series: [
      { values: padTo(f.values, direct.length), color: COLOR.f, label: labels.f },
      { values: padTo(g.values, direct.length), color: COLOR.g, label: labels.g },
      { values: direct.values, color: COLOR.direct, label: labels.direct },
      { values: viaFft.result, color: COLOR.fft, label: labels.fft, dashed: true },
    ],
    subtitle: labels.recipe,
    caption: `${labels.deviation}: ${worst.toExponential(2)}`,
  };
}

// ─── helpers ──────────────────────────────────────────────────────────────────

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Right-pad (or truncate) a series to `length` with zeros. */
function padTo(values: ArrayLike<number>, length: number): Float64Array {
  const out = new Float64Array(length);
  const n = Math.min(values.length, length);
  for (let i = 0; i < n; i++) out[i] = values[i];
  return out;
}

/**
 * Flip the kernel g end-to-end and translate it so its centre sits at `pos`,
 * rendered across `outLength` samples. This mirrors the g[n−m] term during a
 * convolution sweep so the flipped kernel is visible at the current position.
 */
function flippedKernel(g: CoreSignal, pos: number, outLength: number): Float64Array {
  const out = new Float64Array(outLength);
  const center = (g.length - 1) / 2;
  for (let i = 0; i < g.length; i++) {
    const flippedIndex = g.length - 1 - i;
    const idx = Math.round(pos + (i - center));
    if (idx >= 0 && idx < outLength) out[idx] = g.values[flippedIndex];
  }
  return out;
}

/** Translate (NOT flip) the kernel so its centre sits at `pos`. */
function translatedKernel(g: CoreSignal, pos: number, outLength: number): Float64Array {
  const out = new Float64Array(outLength);
  const center = (g.length - 1) / 2;
  for (let i = 0; i < g.length; i++) {
    const idx = Math.round(pos + (i - center));
    if (idx >= 0 && idx < outLength) out[idx] = g.values[i];
  }
  return out;
}

/** DFT cosine coefficient for bin k (real part of F[k]). */
function dftCosine(values: ArrayLike<number>, k: number): number {
  const n = values.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += values[i] * Math.cos((2 * Math.PI * k * i) / n);
  return sum;
}

/** DFT sine coefficient for bin k (negative imaginary part of F[k]). */
function dftSine(values: ArrayLike<number>, k: number): number {
  const n = values.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += values[i] * Math.sin((2 * Math.PI * k * i) / n);
  return sum;
}
