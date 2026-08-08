/**
 * Discrete signal primitives and preset generators.
 *
 * A {@link Signal} is a real-valued sequence sampled on a uniform grid of
 * `length` points. Presets are parameterised generator functions so the UI can
 * expose a small set of knobs (width, frequency, decay …) rather than freehand
 * drawing. All generators are centred around the middle sample so that two
 * symmetric kernels convolve/correlate identically — a fact the lab leans on.
 */

export interface Signal {
  /** Sample values, `values.length === length`. */
  values: Float64Array;
  /** Number of samples. */
  length: number;
}

export type SignalPresetId =
  | "delta"
  | "box"
  | "gaussian"
  | "triangle"
  | "expDecay"
  | "sinc"
  | "cosine"
  | "sine";

/** Identifier of a preset parameter exposed to the UI. */
export type SignalParamId = "width" | "sigma" | "tau" | "frequency";

export interface SignalParamSpec {
  id: SignalParamId;
  /** Localised label is resolved by the React layer; this is just a stable key. */
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface SignalPresetSpec {
  id: SignalPresetId;
  params: SignalParamSpec[];
}

export const SIGNAL_PRESETS: SignalPresetSpec[] = [
  { id: "delta", params: [] },
  {
    id: "box",
    params: [{ id: "width", min: 1, max: 21, step: 2, default: 5 }],
  },
  {
    id: "gaussian",
    params: [{ id: "sigma", min: 0.5, max: 6, step: 0.1, default: 1.5 }],
  },
  {
    id: "triangle",
    params: [{ id: "width", min: 3, max: 21, step: 2, default: 7 }],
  },
  {
    id: "expDecay",
    params: [{ id: "tau", min: 0.5, max: 8, step: 0.1, default: 2 }],
  },
  {
    id: "sinc",
    params: [{ id: "width", min: 1, max: 12, step: 0.5, default: 3 }],
  },
  {
    id: "cosine",
    params: [{ id: "frequency", min: 0.5, max: 8, step: 0.25, default: 2 }],
  },
  {
    id: "sine",
    params: [{ id: "frequency", min: 0.5, max: 8, step: 0.25, default: 2 }],
  },
];

export const SIGNAL_PRESET_IDS: SignalPresetId[] = SIGNAL_PRESETS.map((p) => p.id);

export function getSignalPresetSpec(id: SignalPresetId): SignalPresetSpec {
  const spec = SIGNAL_PRESETS.find((p) => p.id === id);
  if (!spec) throw new Error(`Unknown signal preset: ${id}`);
  return spec;
}

export type SignalParams = Partial<Record<SignalParamId, number>>;

export function defaultSignalParams(id: SignalPresetId): SignalParams {
  const params: SignalParams = {};
  for (const spec of getSignalPresetSpec(id).params) params[spec.id] = spec.default;
  return params;
}

export function makeSignal(values: ArrayLike<number>): Signal {
  const arr = new Float64Array(values.length);
  arr.set(values);
  return { values: arr, length: arr.length };
}

export function zeros(length: number): Signal {
  return { values: new Float64Array(length), length };
}

/**
 * Build a {@link Signal} for the given preset/params at the requested length.
 * The signal is centred on the middle sample so symmetric kernels stay symmetric.
 */
export function buildSignal(
  id: SignalPresetId,
  params: SignalParams,
  length: number,
): Signal {
  const out = new Float64Array(length);
  const center = (length - 1) / 2;
  switch (id) {
    case "delta": {
      if (length > 0) out[Math.round(center)] = 1;
      break;
    }
    case "box": {
      const w = clampInt(params.width ?? 5, 1, length);
      const half = Math.floor(w / 2);
      const amp = 1 / w;
      for (let i = 0; i < length; i++) {
        if (Math.abs(i - center) <= half) out[i] = amp;
      }
      break;
    }
    case "gaussian": {
      const sigma = Math.max(1e-3, params.sigma ?? 1.5);
      let sum = 0;
      for (let i = 0; i < length; i++) {
        const d = (i - center) / sigma;
        const v = Math.exp(-0.5 * d * d);
        out[i] = v;
        sum += v;
      }
      if (sum > 0) for (let i = 0; i < length; i++) out[i] /= sum;
      break;
    }
    case "triangle": {
      const w = clampInt(params.width ?? 7, 1, length);
      const half = (w - 1) / 2;
      let sum = 0;
      for (let i = 0; i < length; i++) {
        const d = Math.abs(i - center);
        const v = d <= half ? 1 - d / (half || 1) : 0;
        out[i] = v;
        sum += v;
      }
      if (sum > 0) for (let i = 0; i < length; i++) out[i] /= sum;
      break;
    }
    case "expDecay": {
      const tau = Math.max(1e-3, params.tau ?? 2);
      let sum = 0;
      for (let i = 0; i < length; i++) {
        const t = i - center;
        // Two-sided exponential: decays away from the centre in both directions.
        const v = Math.exp(-Math.abs(t) / tau);
        out[i] = v;
        sum += v;
      }
      if (sum > 0) for (let i = 0; i < length; i++) out[i] /= sum;
      break;
    }
    case "sinc": {
      const w = Math.max(0.5, params.width ?? 3);
      let sum = 0;
      for (let i = 0; i < length; i++) {
        const x = (i - center) / w;
        const v = Math.abs(x) < 1e-9 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
        out[i] = v;
        sum += v;
      }
      // Sinc is not strictly positive; normalise by peak rather than sum.
      const peak = Math.max(1e-9, ...out);
      for (let i = 0; i < length; i++) out[i] /= peak;
      break;
    }
    case "cosine": {
      const freq = Math.max(1e-3, params.frequency ?? 2);
      for (let i = 0; i < length; i++) {
        out[i] = Math.cos((2 * Math.PI * freq * (i - center)) / length);
      }
      break;
    }
    case "sine": {
      const freq = Math.max(1e-3, params.frequency ?? 2);
      for (let i = 0; i < length; i++) {
        out[i] = Math.sin((2 * Math.PI * freq * (i - center)) / length);
      }
      break;
    }
  }
  return { values: out, length };
}

function clampInt(value: number, min: number, max: number): number {
  const v = Math.round(value);
  if (Number.isNaN(v)) return min;
  return Math.min(max, Math.max(min, v));
}

/** Element-wise sum, result length = max of the inputs (missing samples = 0). */
export function addSignals(a: Signal, b: Signal): Signal {
  const length = Math.max(a.length, b.length);
  const out = new Float64Array(length);
  for (let i = 0; i < length; i++) out[i] = (a.values[i] ?? 0) + (b.values[i] ?? 0);
  return { values: out, length };
}
