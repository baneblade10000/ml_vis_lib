import { createRng, randn, shuffleIndices } from "./rng";
import type { DatasetId, PlaygroundConfig, Sample } from "./types";

function makeCircleDataset(rng: () => number, count: number, noise: number): { points: number[][]; labels: number[] } {
  const innerCount = Math.floor(count / 2);
  const outerCount = count - innerCount;
  const points: number[][] = [];
  const labels: number[] = [];

  for (let i = 0; i < innerCount; i++) {
    const theta = rng() * 2 * Math.PI;
    const radius = 0.42 + randn(rng) * noise;
    points.push([radius * Math.cos(theta), radius * Math.sin(theta)]);
    labels.push(0);
  }
  for (let i = 0; i < outerCount; i++) {
    const theta = rng() * 2 * Math.PI;
    const radius = 0.95 + randn(rng) * noise;
    points.push([radius * Math.cos(theta), radius * Math.sin(theta)]);
    labels.push(1);
  }
  return { points, labels };
}

function makeSpiralDataset(rng: () => number, count: number, noise: number): { points: number[][]; labels: number[] } {
  const firstCount = Math.floor(count / 2);
  const secondCount = count - firstCount;
  const points: number[][] = [];
  const labels: number[] = [];

  for (const [classId, classCount] of [[0, firstCount], [1, secondCount]] as const) {
    for (let i = 0; i < classCount; i++) {
      const t = classCount <= 1 ? 0 : i / (classCount - 1);
      const radius = 0.12 + t * 0.88;
      const theta = classId * Math.PI + 4.2 * radius + randn(rng) * noise * 2;
      points.push([
        radius * Math.cos(theta) + randn(rng) * noise * 0.35,
        radius * Math.sin(theta) + randn(rng) * noise * 0.35,
      ]);
      labels.push(classId);
    }
  }
  return { points, labels };
}

function makeGaussianDataset(rng: () => number, count: number, noise: number): { points: number[][]; labels: number[] } {
  const firstCount = Math.floor(count / 2);
  const secondCount = count - firstCount;
  const spread = 0.24 + noise;
  const points: number[][] = [];
  const labels: number[] = [];

  for (let i = 0; i < firstCount; i++) {
    points.push([-0.45 + randn(rng) * spread, -0.25 + randn(rng) * spread]);
    labels.push(0);
  }
  for (let i = 0; i < secondCount; i++) {
    points.push([0.45 + randn(rng) * spread, 0.3 + randn(rng) * spread]);
    labels.push(1);
  }
  return { points, labels };
}

function makeXorDataset(rng: () => number, count: number, noise: number): { points: number[][]; labels: number[] } {
  const points: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < count; i++) {
    const x1 = rng() * 2 - 1 + randn(rng) * noise;
    const x2 = rng() * 2 - 1 + randn(rng) * noise;
    points.push([x1, x2]);
    labels.push(x1 * x2 < 0 ? 1 : 0);
  }
  return { points, labels };
}

function makeRawDataset(dataset: DatasetId, rng: () => number, count: number, noise: number) {
  switch (dataset) {
    case "xor":
      return makeXorDataset(rng, count, noise);
    case "spiral":
      return makeSpiralDataset(rng, count, noise);
    case "gaussian":
      return makeGaussianDataset(rng, count, noise);
    default:
      return makeCircleDataset(rng, count, noise);
  }
}

export function makeDataset(config: Pick<PlaygroundConfig, "dataset" | "sampleCount" | "noise" | "seed">): Sample[] {
  const rng = createRng(config.seed);
  const count = Math.max(16, config.sampleCount);
  const noise = Math.max(0, config.noise);
  const dataset = config.dataset;
  const { points, labels } = makeRawDataset(dataset, rng, count, noise);
  const order = shuffleIndices(labels.length, rng);
  return order.map((index) => ({
    x1: points[index][0],
    x2: points[index][1],
    target: labels[index],
  }));
}

export function gridPoints(gridSize: number): Array<{ x1: number; x2: number }> {
  const size = Math.max(12, gridSize);
  const axis: number[] = [];
  const min = -1.35;
  const max = 1.35;
  for (let i = 0; i < size; i++) {
    axis.push(min + ((max - min) * i) / (size - 1));
  }
  const points: Array<{ x1: number; x2: number }> = [];
  for (const x2 of axis) {
    for (const x1 of axis) {
      points.push({ x1, x2 });
    }
  }
  return points;
}
