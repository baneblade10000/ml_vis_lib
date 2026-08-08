/**
 * Radix-2 Cooley–Tukey FFT and supporting helpers.
 *
 * The lab uses FFT to (a) show signal spectra and (b) demonstrate the
 * convolution theorem, so besides the core transform we expose magnitude,
 * inverse, and a linear-convolution-via-FFT that matches {@link convolve}.
 */

export interface ComplexArray {
  /** Real parts. */
  re: Float64Array;
  /** Imaginary parts. */
  im: Float64Array;
  /** Number of complex samples (`re.length === im.length === length`). */
  length: number;
}

export function nextPow2(n: number): number {
  if (n <= 1) return 1;
  return 1 << (32 - Math.clz32(n - 1));
}

export function makeComplex(length: number): ComplexArray {
  return { re: new Float64Array(length), im: new Float64Array(length), length };
}

/** Copy a real signal into a complex array, optionally zero-padding to `length`. */
export function toComplex(values: ArrayLike<number>, length?: number): ComplexArray {
  const n = length ?? values.length;
  const re = new Float64Array(n);
  re.set(values);
  return { re, im: new Float64Array(n), length: n };
}

/**
 * In-place iterative radix-2 FFT. `re`/`im` must have the same length, which
 * must be a power of two. Pass `inverse = true` for the inverse transform;
 * the result is already scaled by 1/N, so `ifft(fft(x)) === x`.
 */
export function fftRadix2(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  if (n <= 1) return;
  if (im.length !== n) throw new Error("fftRadix2: re/im length mismatch");
  if (n & (n - 1)) throw new Error("fftRadix2: length must be a power of two");

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (sign * Math.PI) / half;
    const wr = Math.cos(angle);
    const wi = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < half; k++) {
        const ar = re[i + k];
        const ai = im[i + k];
        const br = re[i + k + half];
        const bi = im[i + k + half];
        const tr = br * cr - bi * ci;
        const ti = br * ci + bi * cr;
        re[i + k] = ar + tr;
        im[i + k] = ai + ti;
        re[i + k + half] = ar - tr;
        im[i + k + half] = ai - ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }

  if (inverse) {
    const inv = 1 / n;
    for (let i = 0; i < n; i++) {
      re[i] *= inv;
      im[i] *= inv;
    }
  }
}

/** Forward FFT of a real signal, zero-padded to a power of two ≥ `padTo`. */
export function fft(values: ArrayLike<number>, padTo?: number): ComplexArray {
  const n = nextPow2(padTo ?? values.length);
  const c = toComplex(values, n);
  fftRadix2(c.re, c.im, false);
  return c;
}

/** Inverse FFT of a complex spectrum back to a real signal (imaginary discarded). */
export function ifft(spectrum: ComplexArray): Float64Array {
  const re = Float64Array.from(spectrum.re);
  const im = Float64Array.from(spectrum.im);
  fftRadix2(re, im, true);
  return re.subarray(0, spectrum.length);
}

export interface Spectrum {
  /** Bin magnitudes `|F[k]|`, length `n`. */
  magnitudes: Float64Array;
  /** Bin phases in radians, length `n`. */
  phases: Float64Array;
  /** Number of bins (= FFT size). */
  length: number;
}

/** Magnitude/phase spectrum of a real signal, zero-padded to a power of two. */
export function dftMagnitude(values: ArrayLike<number>, padTo?: number): Spectrum {
  const c = fft(values, padTo);
  const n = c.length;
  const magnitudes = new Float64Array(n);
  const phases = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const re = c.re[i];
    const im = c.im[i];
    magnitudes[i] = Math.hypot(re, im);
    phases[i] = Math.atan2(im, re);
  }
  return { magnitudes, phases, length: n };
}

export interface ConvolutionViaFftResult {
  /** Linear convolution result, length `f.length + g.length − 1`. */
  result: Float64Array;
  /** FFT size used (power of two ≥ result length). */
  fftSize: number;
}

/**
 * Linear convolution via the convolution theorem: `ifft(fft(f) · fft(g))`.
 * Both signals are zero-padded to a power of two ≥ N + M − 1 so circular
 * aliasing never corrupts the linear result. Output is trimmed to N + M − 1.
 */
export function convolveViaFft(f: ArrayLike<number>, g: ArrayLike<number>): ConvolutionViaFftResult {
  const nf = f.length;
  const ng = g.length;
  if (nf === 0 || ng === 0) return { result: new Float64Array(0), fftSize: 1 };
  const outLen = nf + ng - 1;
  const fftSize = nextPow2(outLen);

  const fr = new Float64Array(fftSize);
  const fi = new Float64Array(fftSize);
  fr.set(f);
  const gr = new Float64Array(fftSize);
  const gi = new Float64Array(fftSize);
  gr.set(g);

  fftRadix2(fr, fi, false);
  fftRadix2(gr, gi, false);

  for (let i = 0; i < fftSize; i++) {
    const ar = fr[i];
    const ai = fi[i];
    const br = gr[i];
    const bi = gi[i];
    // complex multiply
    fr[i] = ar * br - ai * bi;
    fi[i] = ar * bi + ai * br;
  }

  fftRadix2(fr, fi, true);
  return { result: fr.subarray(0, outLen), fftSize };
}
