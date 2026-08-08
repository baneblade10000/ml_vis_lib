/**
 * Linear (non-circular) convolution and cross-correlation of two real signals.
 *
 * Convolution flips the kernel, cross-correlation does not — this single flip is
 * the whole point of the lab's first two tabs, so both are kept explicit and use
 * the same output length (N + M − 1).
 */

import type { Signal } from "./signal";

export type { Signal };

/**
 * Linear convolution `y = f ∗ g` with `y[n] = Σ_m f[m] · g[n − m]`.
 * Output length is `f.length + g.length − 1`.
 */
export function convolve(f: Signal, g: Signal): Signal {
  const n = f.length;
  const m = g.length;
  if (n === 0 || m === 0) return { values: new Float64Array(0), length: 0 };
  const out = new Float64Array(n + m - 1);
  for (let i = 0; i < n; i++) {
    const fv = f.values[i];
    if (fv === 0) continue;
    for (let j = 0; j < m; j++) {
      out[i + j] += fv * g.values[j];
    }
  }
  return { values: out, length: out.length };
}

/**
 * Cross-correlation `y[n] = Σ_m f[m] · g[m − n + shift]` (kernel NOT flipped).
 * Output length matches {@link convolve}. `shift` aligns indices so that, for a
 * symmetric kernel, the correlation peak sits at the same centre as a
 * convolution — making the two directly comparable on one chart.
 */
export function correlate(f: Signal, g: Signal): Signal {
  const n = f.length;
  const m = g.length;
  if (n === 0 || m === 0) return { values: new Float64Array(0), length: 0 };
  const out = new Float64Array(n + m - 1);
  const shift = m - 1;
  for (let i = 0; i < n; i++) {
    const fv = f.values[i];
    if (fv === 0) continue;
    for (let j = 0; j < m; j++) {
      // out index = i - j + shift
      out[i - j + shift] += fv * g.values[j];
    }
  }
  return { values: out, length: out.length };
}
