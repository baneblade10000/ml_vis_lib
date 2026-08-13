/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * A two dimensional example: x and y coordinates with the label.
 */
export type Example2D = {
  x: number;
  y: number;
  label: number;
};

type Point = {
  x: number;
  y: number;
};

export const NUM_SAMPLES = 500;

export type Dataset2DClassificationId = "circle" | "xor" | "gauss" | "spiral";
export type Dataset2DRegressionId = "sinSin";
export type DatasetId = Dataset2DClassificationId | Dataset2DRegressionId;
export type DataGenerator = (numSamples: number, noise: number) => Example2D[];

export const DATASETS_2D_CLASSIFICATION: Record<Dataset2DClassificationId, DataGenerator> = {
  circle: classifyCircleData,
  xor: classifyXORData,
  gauss: classifyTwoGaussData,
  spiral: classifySpiralData,
};

export const DATASETS_2D_REGRESSION: Record<Dataset2DRegressionId, DataGenerator> = {
  sinSin: regressSinSin,
};

export const DATASETS: Record<DatasetId, DataGenerator> = {
  ...DATASETS_2D_CLASSIFICATION,
  ...DATASETS_2D_REGRESSION,
};

export const DEFAULT_DATASET_2D_CLASSIFICATION: Dataset2DClassificationId = "circle";
export const DEFAULT_DATASET_2D_REGRESSION: Dataset2DRegressionId = "sinSin";

export function isDataset2DClassificationId(id: string): id is Dataset2DClassificationId {
  return id in DATASETS_2D_CLASSIFICATION;
}

export function isDataset2DRegressionId(id: string): id is Dataset2DRegressionId {
  return id in DATASETS_2D_REGRESSION;
}

/**
 * Shuffles the array using Fisher-Yates algorithm.
 */
export function shuffle(array: Example2D[]): void {
  let counter = array.length;
  while (counter > 0) {
    const index = Math.floor(Math.random() * counter);
    counter--;
    const temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

function linearScale(domain: [number, number], range: [number, number], value: number): number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  if (d1 === d0) return r0;
  const t = (value - d0) / (d1 - d0);
  return r0 + t * (r1 - r0);
}

export function classifyTwoGaussData(numSamples: number, noise: number): Example2D[] {
  const points: Example2D[] = [];
  const variance = linearScale([0, 0.5], [0.5, 4], noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      const x = normalRandom(cx, variance);
      const y = normalRandom(cy, variance);
      points.push({ x, y, label });
    }
  }

  genGauss(2, 2, 1);
  genGauss(-2, -2, -1);
  return points;
}

export function classifySpiralData(numSamples: number, noise: number): Example2D[] {
  const points: Example2D[] = [];
  const n = numSamples / 2;

  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      const r = (i / n) * 5;
      const t = (1.75 * i / n) * 2 * Math.PI + deltaT;
      const x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      const y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({ x, y, label });
    }
  }

  genSpiral(0, 1);
  genSpiral(Math.PI, -1);
  return points;
}

export function classifyCircleData(numSamples: number, noise: number): Example2D[] {
  const points: Example2D[] = [];
  const radius = 5;
  function getCircleLabel(p: Point, center: Point) {
    return dist(p, center) < radius * 0.5 ? 1 : -1;
  }

  for (let i = 0; i < numSamples / 2; i++) {
    const r = randUniform(0, radius * 0.5);
    const angle = randUniform(0, 2 * Math.PI);
    const x = r * Math.sin(angle);
    const y = r * Math.cos(angle);
    const noiseX = randUniform(-radius, radius) * noise;
    const noiseY = randUniform(-radius, radius) * noise;
    const label = getCircleLabel({ x: x + noiseX, y: y + noiseY }, { x: 0, y: 0 });
    points.push({ x, y, label });
  }

  for (let i = 0; i < numSamples / 2; i++) {
    const r = randUniform(radius * 0.7, radius);
    const angle = randUniform(0, 2 * Math.PI);
    const x = r * Math.sin(angle);
    const y = r * Math.cos(angle);
    const noiseX = randUniform(-radius, radius) * noise;
    const noiseY = randUniform(-radius, radius) * noise;
    const label = getCircleLabel({ x: x + noiseX, y: y + noiseY }, { x: 0, y: 0 });
    points.push({ x, y, label });
  }
  return points;
}

export function classifyXORData(numSamples: number, noise: number): Example2D[] {
  function getXORLabel(p: Point) {
    return p.x * p.y >= 0 ? 1 : -1;
  }

  const points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-5, 5);
    const padding = 0.3;
    x += x > 0 ? padding : -padding;
    let y = randUniform(-5, 5);
    y += y > 0 ? padding : -padding;
    const noiseX = randUniform(-5, 5) * noise;
    const noiseY = randUniform(-5, 5) * noise;
    const label = getXORLabel({ x: x + noiseX, y: y + noiseY });
    points.push({ x, y, label });
  }
  return points;
}

/**
 * Continuous target sin(x)·sin(y) ∈ [-1, 1].
 * Points are laid on a grid so the checkerboard pattern stays readable;
 * label noise is applied only when the noise slider is > 0.
 */
export function regressSinSin(numSamples: number, noise: number): Example2D[] {
  const points: Example2D[] = [];
  const side = Math.max(2, Math.round(Math.sqrt(numSamples)));
  const lo = -6;
  const hi = 6;
  const span = hi - lo;
  for (let i = 0; i < side; i++) {
    for (let j = 0; j < side; j++) {
      // Cell centers — avoids clustering on the domain border.
      const x = lo + ((i + 0.5) / side) * span;
      const y = lo + ((j + 0.5) / side) * span;
      let label = Math.sin(x) * Math.sin(y);
      if (noise > 0) {
        label += randUniform(-1, 1) * noise * 0.5;
      }
      points.push({ x, y, label });
    }
  }
  return points;
}

function randUniform(a: number, b: number) {
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

  const result = Math.sqrt((-2 * Math.log(s)) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}

function dist(a: Point, b: Point): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}
