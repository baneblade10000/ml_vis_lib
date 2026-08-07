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

import { valueToRgb } from "../playground/network/colors";

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

export type ValueMatrixLayout = "col-major" | "row-major";
/** `diverging` = TF Playground violet→magenta; `gray` = bright black→white. */
export type ValueMatrixPalette = "diverging" | "gray";

export interface RenderValueMatrixOptions {
  discretize?: boolean;
  alpha?: number;
  layout?: ValueMatrixLayout;
  palette?: ValueMatrixPalette;
  /**
   * Stretch the matrix into the full palette range using its own min/max.
   * Defaults to true for `gray`, false for `diverging`.
   */
  autoscale?: boolean;
}

function sample(
  matrix: number[][],
  x: number,
  y: number,
  layout: ValueMatrixLayout,
): number {
  return layout === "row-major" ? matrix[y]![x]! : matrix[x]![y]!;
}

function matrixRange(
  matrix: number[][],
  width: number,
  height: number,
  layout: ValueMatrixLayout,
): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = sample(matrix, x, y, layout);
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 };
  if (max - min < 1e-9) return { min: min - 0.5, max: max + 0.5 };
  return { min, max };
}

/**
 * Render a value matrix to a canvas using ImageData.
 *
 * Positional form (TF Playground): `(canvas, matrix, discretize?, alpha?, layout?)`
 * Options form: `(canvas, matrix, { layout: "row-major", palette: "gray" })`
 */
export function renderValueMatrix(
  canvas: HTMLCanvasElement,
  matrix: number[][],
  discretizeOrOptions: boolean | RenderValueMatrixOptions = false,
  alphaArg = 160,
  layoutArg: ValueMatrixLayout = "col-major",
): void {
  const opts: RenderValueMatrixOptions =
    typeof discretizeOrOptions === "object" && discretizeOrOptions !== null
      ? discretizeOrOptions
      : {
          discretize: discretizeOrOptions,
          alpha: alphaArg,
          layout: layoutArg,
        };

  const layout = opts.layout ?? "col-major";
  const palette = opts.palette ?? "diverging";
  const discretize = opts.discretize ?? false;
  // Diverging heatmaps are opaque so the white node chrome does not wash
  // midtones into a pale fog (TF Playground also paints solid shades).
  const alpha = opts.alpha ?? 255;
  const autoscale = opts.autoscale ?? palette === "gray";

  const outer = matrix.length;
  const inner = matrix[0]?.length ?? 0;
  if (!outer || !inner) return;

  const width = layout === "row-major" ? inner : outer;
  const height = layout === "row-major" ? outer : inner;
  if (layout === "col-major" && width !== height) return;

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    imageDataCache.delete(canvas);
  }
  const context = canvas.getContext("2d");
  if (!context) return;

  let cached = imageDataCache.get(canvas);
  if (!cached || cached.w !== width || cached.h !== height) {
    cached = { w: width, h: height, image: context.createImageData(width, height) };
    imageDataCache.set(canvas, cached);
  }
  const image = cached.image;
  const range = autoscale ? matrixRange(matrix, width, height, layout) : { min: -1, max: 1 };
  const span = range.max - range.min;

  for (let y = 0, p = -1; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = sample(matrix, x, y, layout);
      if (discretize) {
        value = value >= 0 ? 1 : -1;
      }

      let r: number;
      let g: number;
      let b: number;
      if (palette === "gray") {
        const t = Math.min(1, Math.max(0, (value - range.min) / span));
        // Slight gamma so mid-tones stay punchy on small upscaled tiles.
        const g8 = Math.round(Math.pow(t, 0.85) * 255);
        r = g = b = g8;
      } else {
        const mapped = autoscale ? ((value - range.min) / span) * 2 - 1 : value;
        ({ r, g, b } = valueToRgb(mapped));
      }

      image.data[++p] = r;
      image.data[++p] = g;
      image.data[++p] = b;
      image.data[++p] = alpha;
    }
  }
  context.putImageData(image, 0, 0);
}
