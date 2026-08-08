import { describe, expect, it } from "vitest";
import {
  buildSignal,
  convolve,
  convolveViaFft,
  correlate,
  fftRadix2,
  ifft,
  makeSignal,
  nextPow2,
  toComplex,
} from "./index";

const maxDev = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  const n = Math.min(a.length, b.length);
  let worst = 0;
  for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
};

describe("convolve", () => {
  it("is identity when convolving with a unit impulse", () => {
    const f = makeSignal([1, 2, 3, 4, 5]);
    const delta = makeSignal([0, 0, 1, 0, 0]);
    const y = convolve(f, delta);
    expect(y.length).toBe(f.length + delta.length - 1);
    // Peak of the impulse sits at index 2, so f is reproduced at offset 2.
    expect(Array.from(y.values.slice(2, 7))).toEqual([1, 2, 3, 4, 5]);
  });

  it("flips the kernel (1-2-1 averaged over a step gives a ramp)", () => {
    const f = makeSignal([0, 0, 1, 1, 1, 1, 0, 0]);
    const g = makeSignal([1, 2, 1]);
    const y = convolve(f, g);
    // Manual: the flipped kernel [1,2,1] sweeps the rising then falling edge.
    expect(y.length).toBe(f.length + g.length - 1);
    expect(Array.from(y.values)).toEqual([0, 0, 1, 3, 4, 4, 3, 1, 0, 0]);
  });
});

describe("correlate", () => {
  it("matches convolution for a symmetric kernel", () => {
    const f = buildSignal("cosine", { frequency: 1.5 }, 32);
    const g = buildSignal("gaussian", { sigma: 2 }, 32);
    const conv = convolve(f, g);
    const corr = correlate(f, g);
    expect(corr.length).toBe(conv.length);
    expect(maxDev(conv.values, corr.values)).toBeLessThan(1e-12);
  });

  it("differs from convolution for an asymmetric kernel", () => {
    const f = makeSignal([0, 0, 0, 1, 0, 0, 0]);
    const g = makeSignal([0, 1, 2]); // asymmetric
    const conv = convolve(f, g);
    const corr = correlate(f, g);
    expect(maxDev(conv.values, corr.values)).toBeGreaterThan(0.1);
  });

  it("peaks where the pattern matches the signal", () => {
    const pattern = makeSignal([1, 2, 1]);
    const signal = makeSignal([0, 0, 1, 2, 1, 0, 0]);
    const corr = correlate(signal, pattern);
    const peakIndex = corr.values.indexOf(Math.max(...corr.values));
    // Correlation of a signal with its own embedded pattern peaks at its centre.
    expect(peakIndex).toBeGreaterThanOrEqual(2);
    expect(peakIndex).toBeLessThanOrEqual(4);
  });
});

describe("fft", () => {
  it("nextPow2 rounds up to powers of two", () => {
    expect(nextPow2(1)).toBe(1);
    expect(nextPow2(2)).toBe(2);
    expect(nextPow2(3)).toBe(4);
    expect(nextPow2(5)).toBe(8);
    expect(nextPow2(8)).toBe(8);
    expect(nextPow2(9)).toBe(16);
    expect(nextPow2(33)).toBe(64);
  });

  it("ifft(fft(x)) recovers x", () => {
    const x = [0.1, 0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.1];
    const c = toComplex(x, nextPow2(x.length));
    fftRadix2(c.re, c.im, false);
    const back = ifft({ ...c, length: c.length });
    expect(maxDev(back, x)).toBeLessThan(1e-12);
  });

  it("places a pure sinusoid's energy in the matching bins", () => {
    const n = 64;
    const k = 4;
    const re = new Float64Array(n);
    for (let i = 0; i < n; i++) re[i] = Math.cos((2 * Math.PI * k * i) / n);
    const im = new Float64Array(n);
    fftRadix2(re, im, false);
    // Magnitude is concentrated in bins k and n-k; a pure cosine contributes
    // exactly n/2 to each of those two bins and zero elsewhere.
    const mag = (i: number) => Math.hypot(re[i], im[i]);
    expect(mag(k)).toBeCloseTo(n / 2, 6);
    expect(mag(n - k)).toBeCloseTo(n / 2, 6);
    for (let i = 0; i < n; i++) {
      if (i !== k && i !== n - k) expect(mag(i)).toBeLessThan(1e-9);
    }
  });
});

describe("convolveViaFft", () => {
  it("matches the direct linear convolution", () => {
    const f = buildSignal("box", { width: 5 }, 24);
    const g = buildSignal("expDecay", { tau: 1.5 }, 24);
    const direct = convolve(f, g);
    const viaFft = convolveViaFft(f.values, g.values);
    expect(viaFft.result.length).toBe(direct.length);
    expect(maxDev(direct.values, viaFft.result)).toBeLessThan(1e-9);
  });

  it("matches direct convolution for arbitrary asymmetric inputs", () => {
    const f = [0, 0, 0, 0, 1, 2, 3, 0, 0, 0];
    const g = [1, -1, 0.5];
    const direct = convolve(makeSignal(f), makeSignal(g));
    const viaFft = convolveViaFft(f, g);
    expect(viaFft.result.length).toBe(direct.length);
    expect(maxDev(direct.values, viaFft.result)).toBeLessThan(1e-9);
  });
});
