/**
 * Tensor data structures for the convolutional-network engine.
 *
 * A {@link Map2D} is a single channel — a `h × w` matrix indexed as
 * `map[row][col]` (row 0 = top). A {@link Volume} stacks channels: `volume[c]`
 * is a `Map2D`. A {@link Signal} is the 1-D analogue: `signal[c]` is a length-`L`
 * row of values.
 */

/** One channel of a 2-D feature map: `map[row][col]`. */
export type Map2D = number[][];

/** A multi-channel 2-D activation: `volume[channel][row][col]`. */
export type Volume = Map2D[];

/** A multi-channel 1-D signal: `signal[channel][position]`. */
export type Signal = number[][];

/** A 2-D kernel bank: `kernels[outChannel][inChannel][row][col]`. */
export type Kernel2D = Map2D[][];

/** A 1-D kernel bank: `kernels[outChannel][inChannel][position]`. */
export type Kernel1D = number[][][];

export type Spatial = { rows: number; cols: number };

export type VolumeShape = { channels: number; rows: number; cols: number };
export type SignalShape = { channels: number; length: number };

// ─── 2-D helpers ──────────────────────────────────────────────────────────────

export function zeros2D(rows: number, cols: number): Map2D {
  const out: Map2D = new Array(rows);
  for (let r = 0; r < rows; r++) out[r] = new Array(cols).fill(0);
  return out;
}

export function zerosVolume(channels: number, rows: number, cols: number): Volume {
  const out: Volume = new Array(channels);
  for (let c = 0; c < channels; c++) out[c] = zeros2D(rows, cols);
  return out;
}

/** Zero every cell of an existing volume (no reallocation). */
export function clearVolume(volume: Volume): void {
  for (let c = 0; c < volume.length; c++) {
    const ch = volume[c]!;
    for (let r = 0; r < ch.length; r++) ch[r]!.fill(0);
  }
}

/**
 * Reuse `cache` when shape matches (cleared); otherwise allocate a fresh volume.
 * Cuts GC traffic in hot conv forward/backward loops.
 */
export function acquireVolume(
  cache: Volume,
  channels: number,
  rows: number,
  cols: number,
): Volume {
  if (
    cache.length === channels &&
    cache[0]?.length === rows &&
    cache[0]?.[0]?.length === cols
  ) {
    clearVolume(cache);
    return cache;
  }
  return zerosVolume(channels, rows, cols);
}

export function clone2D(map: Map2D): Map2D {
  return map.map((row) => row.slice());
}

export function cloneVolume(volume: Volume): Volume {
  return volume.map(clone2D);
}

export function cloneKernel2D(kernels: Kernel2D): Kernel2D {
  return kernels.map((outBank) => outBank.map(clone2D));
}

/** Apply `fn` elementwise to a 2-D map (in place). */
export function map2DInPlace(map: Map2D, fn: (v: number, r: number, c: number) => number): void {
  for (let r = 0; r < map.length; r++) {
    const row = map[r];
    for (let c = 0; c < row.length; c++) row[c] = fn(row[c], r, c);
  }
}

/** Sum every element of a 2-D map. */
export function sum2D(map: Map2D): number {
  let s = 0;
  for (const row of map) for (const v of row) s += v;
  return s;
}

/** Sum every element of a volume. */
export function sumVolume(volume: Volume): number {
  let s = 0;
  for (const ch of volume) s += sum2D(ch);
  return s;
}

// ─── 1-D helpers ──────────────────────────────────────────────────────────────

export function zeros1D(length: number): number[] {
  return new Array(length).fill(0);
}

export function zerosSignal(channels: number, length: number): Signal {
  const out: Signal = new Array(channels);
  for (let c = 0; c < channels; c++) out[c] = zeros1D(length);
  return out;
}

export function clearSignal(signal: Signal): void {
  for (let c = 0; c < signal.length; c++) signal[c]!.fill(0);
}

export function acquireSignal(cache: Signal, channels: number, length: number): Signal {
  if (cache.length === channels && cache[0]?.length === length) {
    clearSignal(cache);
    return cache;
  }
  return zerosSignal(channels, length);
}

export function clone1D(row: number[]): number[] {
  return row.slice();
}

export function cloneSignal(signal: Signal): Signal {
  return signal.map(clone1D);
}

export function cloneKernel1D(kernels: Kernel1D): Kernel1D {
  return kernels.map((outBank) => outBank.map(clone1D));
}

export function sum1D(row: number[]): number {
  let s = 0;
  for (const v of row) s += v;
  return s;
}

// ─── RNG ──────────────────────────────────────────────────────────────────────

/** `n` independent standard-normal samples via Box–Muller (preserved parity). */
export function gaussian(n: number): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    out[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
  return out;
}

/** Scale He/Xavier-style init to fan-in. Values in roughly `[-bound, bound]`. */
export function randomInit(size: number, fanIn: number): number[] {
  const bound = Math.sqrt(6 / Math.max(fanIn, 1));
  const out = new Array<number>(size);
  for (let i = 0; i < size; i++) out[i] = (Math.random() * 2 - 1) * bound;
  return out;
}
