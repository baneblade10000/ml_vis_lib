/**
 * Lightweight tensor types for gallery datasets / UI.
 * Training tensors live in CNN WASM.
 */

/** One channel of a 2-D feature map: `map[row][col]`. */
export type Map2D = number[][];

/** A multi-channel 2-D activation: `volume[channel][row][col]`. */
export type Volume = Map2D[];

/** A multi-channel 1-D signal: `signal[channel][position]`. */
export type Signal = number[][];

export function zeros2D(rows: number, cols: number): Map2D {
  const out: Map2D = new Array(rows);
  for (let r = 0; r < rows; r++) out[r] = new Array(cols).fill(0);
  return out;
}
