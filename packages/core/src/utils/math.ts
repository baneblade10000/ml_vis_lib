export function extent(values: number[]): [number, number] | null {
  if (values.length === 0) return null;
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return [min, max];
}

export function downsample<T>(items: T[], targetCount: number): T[] {
  if (items.length <= targetCount) return items;
  const step = items.length / targetCount;
  const result: T[] = [];
  for (let i = 0; i < targetCount; i++) {
    result.push(items[Math.floor(i * step)]!);
  }
  return result;
}
