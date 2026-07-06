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

import { valueToRgb } from "../playground/tf/colors";

/** Average each factor×factor block in a square matrix. */
export function reduceMatrix(matrix: number[][], factor: number): number[][] {
  if (matrix.length !== matrix[0].length) {
    throw new Error("The provided matrix must be a square matrix");
  }
  if (matrix.length % factor !== 0) {
    throw new Error(
      "The width/height of the matrix must be divisible by the reduction factor",
    );
  }
  const result: number[][] = new Array(matrix.length / factor);
  for (let i = 0; i < matrix.length; i += factor) {
    result[i / factor] = new Array(matrix.length / factor);
    for (let j = 0; j < matrix.length; j += factor) {
      let avg = 0;
      for (let k = 0; k < factor; k++) {
        for (let l = 0; l < factor; l++) {
          avg += matrix[i + k][j + l];
        }
      }
      avg /= factor * factor;
      result[i / factor][j / factor] = avg;
    }
  }
  return result;
}

const imageDataCache = new WeakMap<
  HTMLCanvasElement,
  { w: number; h: number; image: ImageData }
>();

/**
 * Render a value matrix in [-1, 1] to a canvas using ImageData (TF Playground style).
 * Matrix is indexed as matrix[col][row].
 */
export function renderValueMatrix(
  canvas: HTMLCanvasElement,
  matrix: number[][],
  discretize = false,
  alpha = 160,
): void {
  const dx = matrix.length;
  const dy = matrix[0]?.length ?? 0;
  if (!dx || dy !== dx) return;

  if (canvas.width !== dx || canvas.height !== dy) {
    canvas.width = dx;
    canvas.height = dy;
    imageDataCache.delete(canvas);
  }
  const context = canvas.getContext("2d");
  if (!context) return;

  let cached = imageDataCache.get(canvas);
  if (!cached || cached.w !== dx || cached.h !== dy) {
    cached = { w: dx, h: dy, image: context.createImageData(dx, dy) };
    imageDataCache.set(canvas, cached);
  }
  const image = cached.image;
  for (let y = 0, p = -1; y < dy; y++) {
    for (let x = 0; x < dx; x++) {
      let value = matrix[x][y];
      if (discretize) {
        value = value >= 0 ? 1 : -1;
      }
      const { r, g, b } = valueToRgb(value);
      image.data[++p] = r;
      image.data[++p] = g;
      image.data[++p] = b;
      image.data[++p] = alpha;
    }
  }
  context.putImageData(image, 0, 0);
}
