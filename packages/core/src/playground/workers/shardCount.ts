/**
 * Default number of data-parallel shard workers.
 * Leaves one core for the UI / coordinator rAF when possible.
 */
export function defaultShardCount(explicit?: number): number {
  if (explicit !== undefined && explicit > 0) return Math.floor(explicit);
  const hc =
    typeof navigator !== "undefined" && typeof navigator.hardwareConcurrency === "number"
      ? navigator.hardwareConcurrency
      : 4;
  return Math.max(1, Math.floor(hc) - 1);
}

/** Split `length` indices into `shardCount` nearly equal contiguous ranges. */
export function partitionRanges(
  length: number,
  shardCount: number,
): Array<{ start: number; end: number }> {
  const n = Math.max(1, shardCount);
  if (length <= 0) return Array.from({ length: n }, () => ({ start: 0, end: 0 }));
  const ranges: Array<{ start: number; end: number }> = [];
  for (let s = 0; s < n; s++) {
    const start = Math.floor((s * length) / n);
    const end = Math.floor(((s + 1) * length) / n);
    ranges.push({ start, end });
  }
  return ranges;
}

/** Element-wise sum of same-length grad buffers. */
export function sumGradBuffers(buffers: Float64Array[]): Float64Array {
  if (buffers.length === 0) return new Float64Array(0);
  const out = new Float64Array(buffers[0]!.length);
  for (const buf of buffers) {
    for (let i = 0; i < out.length; i++) out[i]! += buf[i]!;
  }
  return out;
}
