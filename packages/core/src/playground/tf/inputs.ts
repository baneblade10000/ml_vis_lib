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

export interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

export const INPUTS: Record<string, InputFeature> = {
  x: { f: (x, _y) => x, label: "X_1" },
  y: { f: (_x, y) => y, label: "X_2" },
  xSquared: { f: (x, _y) => x * x, label: "X_1^2" },
  ySquared: { f: (_x, y) => y * y, label: "X_2^2" },
  xTimesY: { f: (x, y) => x * y, label: "X_1X_2" },
  sinX: { f: (x, _y) => Math.sin(x), label: "sin(X_1)" },
  sinY: { f: (_x, y) => Math.sin(y), label: "sin(X_2)" },
};

export const INPUT_IDS = Object.keys(INPUTS);

export function constructInputIds(enabledFeatures: Record<string, boolean>): string[] {
  const result: string[] = [];
  for (const inputName of INPUT_IDS) {
    if (enabledFeatures[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

export function constructInput(
  x: number,
  y: number,
  enabledFeatures: Record<string, boolean>,
): number[] {
  const input: number[] = [];
  for (const inputName of INPUT_IDS) {
    if (enabledFeatures[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}
