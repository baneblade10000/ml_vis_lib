import type { FeatureName, Sample } from "./types";

type PointLike = Pick<Sample, "x1" | "x2"> | { x1: number; x2: number };

export function featureMatrix(samples: PointLike[], featureNames: FeatureName[]): number[][] {
  const x1 = samples.map((s) => s.x1);
  const x2 = samples.map((s) => s.x2);
  const valuesByName: Record<FeatureName, number[]> = {
    x1,
    x2,
    x1_squared: x1.map((v) => v * v),
    x2_squared: x2.map((v) => v * v),
    x1_x2: x1.map((v, i) => v * x2[i]),
    sin_x1: x1.map((v) => Math.sin(Math.PI * v)),
    sin_x2: x2.map((v) => Math.sin(Math.PI * v)),
  };

  const selected = featureNames.filter((name) => name in valuesByName);
  const names = selected.length ? selected : (["x1", "x2"] as FeatureName[]);
  const rows = samples.length;
  const matrix: number[][] = Array.from({ length: rows }, () => []);
  for (const name of names) {
    const column = valuesByName[name];
    for (let i = 0; i < rows; i++) matrix[i].push(column[i]);
  }
  return matrix;
}

export function normalizeFeatures(
  features: number[][],
): { normalized: number[][]; mean: number[]; std: number[] } {
  if (!features.length || !features[0].length) {
    return { normalized: features, mean: [], std: [] };
  }
  const dims = features[0].length;
  const mean = Array.from({ length: dims }, () => 0);
  const std = Array.from({ length: dims }, () => 1);

  for (let d = 0; d < dims; d++) {
    let sum = 0;
    for (const row of features) sum += row[d];
    mean[d] = sum / features.length;
  }
  for (let d = 0; d < dims; d++) {
    let variance = 0;
    for (const row of features) {
      const delta = row[d] - mean[d];
      variance += delta * delta;
    }
    std[d] = Math.sqrt(variance / features.length) || 1;
  }

  const normalized = features.map((row) =>
    row.map((value, d) => (value - mean[d]) / std[d]),
  );
  return { normalized, mean, std };
}

export function applyNormalization(
  features: number[][],
  mean: number[],
  std: number[],
): number[][] {
  return features.map((row) => row.map((value, d) => (value - mean[d]) / (std[d] || 1)));
}
