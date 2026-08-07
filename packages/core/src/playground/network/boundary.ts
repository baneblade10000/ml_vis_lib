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

import type { GridPoint } from "../types";
import { constructInput, INPUTS } from "./inputs";
import { DENSITY, X_DOMAIN } from "./constants";
import { forwardProp, forEachNode, type Node } from "./nn";

function scaleLinear(domain: [number, number], range: [number, number], value: number): number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

/**
 * Given a neural network, computes the output of every node on a square grid.
 * Returns a map where each key is the node ID and the value is a square matrix
 * indexed as matrix[col][row] (column-major heatmap convention).
 */
export function computeBoundaries(
  network: Node[][],
  activeInputs: Record<string, boolean>,
  density = DENSITY,
  xDomain: [number, number] = X_DOMAIN,
  includeInputFeatures = true,
): Record<string, number[][]> {
  const boundary: Record<string, number[][]> = {};

  forEachNode(network, true, (node) => {
    boundary[node.id] = Array.from({ length: density }, () => new Array<number>(density));
  });

  if (includeInputFeatures) {
    for (const nodeId of Object.keys(INPUTS)) {
      boundary[nodeId] = Array.from({ length: density }, () => new Array<number>(density));
    }
  }

  const xScale = (i: number) => scaleLinear([0, density - 1], xDomain, i);
  const yScale = (j: number) => scaleLinear([density - 1, 0], xDomain, j);

  for (let i = 0; i < density; i++) {
    for (let j = 0; j < density; j++) {
      const x = xScale(i);
      const y = yScale(j);
      const input = constructInput(x, y, activeInputs);
      forwardProp(network, input);
      forEachNode(network, true, (node) => {
        boundary[node.id][i][j] = node.output;
      });
      if (includeInputFeatures) {
        for (const nodeId of Object.keys(INPUTS)) {
          if (activeInputs[nodeId]) {
            boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
          }
        }
      }
    }
  }

  return boundary;
}

/** Convert boundary matrix to GridPoint[] for DecisionBoundaryPlot. */
export function boundaryToGridPoints(
  matrix: number[][],
  xDomain: [number, number] = [-6, 6],
  mapToProbability = true,
): GridPoint[] {
  const density = matrix.length;
  if (!density || matrix[0]?.length !== density) return [];

  const points: GridPoint[] = [];
  for (let i = 0; i < density; i++) {
    for (let j = 0; j < density; j++) {
      const x1 = scaleLinear([0, density - 1], xDomain, i);
      const x2 = scaleLinear([density - 1, 0], xDomain, j);
      const raw = matrix[i][j];
      const probability = mapToProbability ? (raw + 1) / 2 : raw;
      points.push({ x1, x2, probability });
    }
  }
  return points;
}

/** Convert Example2D samples to DecisionBoundaryPlot Sample format. */
export function examplesToSamples(examples: Array<{ x: number; y: number; label: number }>) {
  return examples.map((e) => ({
    x1: e.x,
    x2: e.y,
    target: e.label,
  }));
}
