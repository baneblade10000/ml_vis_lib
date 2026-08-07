import type { Example2D } from "./dataset";
import { NUM_SAMPLES, shuffle } from "./dataset";
import { X_DOMAIN } from "./constants";

/** Same shape as Example2D: for 1D, `y` mirrors `label` so overlays plot at label height. */
export type Example1D = Example2D;

export type Dataset1DClassificationId = "gauss1d" | "threshold" | "twoClusters";
export type Dataset1DRegressionId = "sine" | "linear" | "cubic" | "step";
export type Dataset1DId = Dataset1DClassificationId | Dataset1DRegressionId;

export type DataGenerator1D = (numSamples: number, noise: number) => Example1D[];

export const DATASETS_1D_CLASSIFICATION: Record<Dataset1DClassificationId, DataGenerator1D> = {
  gauss1d: classifyGauss1D,
  threshold: classifyThreshold1D,
  twoClusters: classifyTwoClusters1D,
};

export const DATASETS_1D_REGRESSION: Record<Dataset1DRegressionId, DataGenerator1D> = {
  sine: regressSine,
  linear: regressLinear,
  cubic: regressCubic,
  step: regressStep,
};

export const DATASETS_1D: Record<Dataset1DId, DataGenerator1D> = {
  ...DATASETS_1D_CLASSIFICATION,
  ...DATASETS_1D_REGRESSION,
};

export const DEFAULT_DATASET_1D_CLASSIFICATION: Dataset1DClassificationId = "gauss1d";
export const DEFAULT_DATASET_1D_REGRESSION: Dataset1DRegressionId = "sine";

export const FEATURES_1D = [
  "x",
  "xSquared",
  "sinX",
] as const;

export const FEATURES_2D_ONLY = [
  "y",
  "ySquared",
  "xTimesY",
  "sinY",
] as const;

export function isDataset1DId(id: string): id is Dataset1DId {
  return id in DATASETS_1D;
}

export function isDataset1DClassificationId(id: string): id is Dataset1DClassificationId {
  return id in DATASETS_1D_CLASSIFICATION;
}

export function isDataset1DRegressionId(id: string): id is Dataset1DRegressionId {
  return id in DATASETS_1D_REGRESSION;
}

/** Ideal target curve for regression datasets (null for classification). */
export function targetCurve1D(
  dataset: Dataset1DId,
  density: number,
  xDomain: [number, number] = X_DOMAIN,
): number[] | null {
  const fn = TARGET_FNS[dataset];
  if (!fn) return null;
  const out = new Array<number>(density);
  for (let i = 0; i < density; i++) {
    const x = xDomain[0] + ((xDomain[1] - xDomain[0]) * i) / (density - 1);
    out[i] = fn(x);
  }
  return out;
}

const TARGET_FNS: Partial<Record<Dataset1DId, (x: number) => number>> = {
  sine: (x) => Math.sin(x),
  linear: (x) => 0.4 * x,
  cubic: (x) => 0.03 * x * x * x - 0.2 * x,
  step: (x) => (x < 0 ? -1 : 1),
};

function point(x: number, label: number): Example1D {
  return { x, y: label, label };
}

function randUniform(a: number, b: number): number {
  return Math.random() * (b - a) + a;
}

function normalRandom(mean = 0, variance = 1): number {
  let v1: number;
  let v2: number;
  let s: number;
  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);
  return mean + Math.sqrt(variance) * Math.sqrt((-2 * Math.log(s)) / s) * v1;
}

function classifyGauss1D(numSamples: number, noise: number): Example1D[] {
  const variance = 0.4 + noise * 2;
  const points: Example1D[] = [];
  for (let i = 0; i < numSamples / 2; i++) {
    points.push(point(normalRandom(-2.5, variance), -1));
    points.push(point(normalRandom(2.5, variance), 1));
  }
  return points;
}

function classifyThreshold1D(numSamples: number, noise: number): Example1D[] {
  const points: Example1D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-6, 6);
    const noisy = x + randUniform(-1, 1) * noise * 3;
    points.push(point(x, noisy < 0 ? -1 : 1));
  }
  return points;
}

function classifyTwoClusters1D(numSamples: number, noise: number): Example1D[] {
  const variance = 0.25 + noise * 1.5;
  const points: Example1D[] = [];
  const n = Math.floor(numSamples / 4);
  for (let i = 0; i < n; i++) {
    points.push(point(normalRandom(-4, variance), -1));
    points.push(point(normalRandom(-1, variance), 1));
    points.push(point(normalRandom(1, variance), -1));
    points.push(point(normalRandom(4, variance), 1));
  }
  return points;
}

function regressWith(
  numSamples: number,
  noise: number,
  fn: (x: number) => number,
  noiseScale: number,
): Example1D[] {
  const points: Example1D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-6, 6);
    const label = fn(x) + randUniform(-1, 1) * noise * noiseScale;
    points.push(point(x, label));
  }
  return points;
}

function regressSine(numSamples: number, noise: number): Example1D[] {
  return regressWith(numSamples, noise, TARGET_FNS.sine!, 0.5);
}

function regressLinear(numSamples: number, noise: number): Example1D[] {
  return regressWith(numSamples, noise, TARGET_FNS.linear!, 0.8);
}

function regressCubic(numSamples: number, noise: number): Example1D[] {
  return regressWith(numSamples, noise, TARGET_FNS.cubic!, 0.6);
}

function regressStep(numSamples: number, noise: number): Example1D[] {
  return regressWith(numSamples, noise, TARGET_FNS.step!, 0.4);
}

/** Generate + shuffle + train/test split for a 1D dataset. */
export function generateSplit1D(
  dataset: Dataset1DId,
  noisePercent: number,
  percTrainData: number,
  numSamples = NUM_SAMPLES,
): { train: Example1D[]; test: Example1D[] } {
  const data = DATASETS_1D[dataset](numSamples, noisePercent / 100);
  shuffle(data);
  const splitIndex = Math.floor((data.length * percTrainData) / 100);
  return { train: data.slice(0, splitIndex), test: data.slice(splitIndex) };
}
