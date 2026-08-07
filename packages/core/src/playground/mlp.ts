import { createRng } from "./rng";
import type { ActivationId, OptimizerId } from "./types";

export interface LinearLayer {
  weights: Float64Array;
  bias: Float64Array;
  inputSize: number;
  outputSize: number;
}

export interface MLP {
  layers: LinearLayer[];
  activation: ActivationId;
  optimizer: OptimizerId;
  learningRate: number;
  adamStep: number;
  adamM: Float64Array;
  adamV: Float64Array;
}

interface ForwardCache {
  inputs: Float64Array[];
  preActivations: Float64Array[];
}

function xavierInit(inputSize: number, outputSize: number, rng: () => number): Float64Array {
  const limit = Math.sqrt(6 / (inputSize + outputSize));
  const weights = new Float64Array(inputSize * outputSize);
  for (let i = 0; i < weights.length; i++) weights[i] = (rng() * 2 - 1) * limit;
  return weights;
}

export function createMLP(
  inputSize: number,
  hiddenLayers: number[],
  activation: ActivationId,
  optimizer: OptimizerId,
  learningRate: number,
  seed: number,
): MLP {
  const rng = createRng(seed + 9001);
  const sizes = [inputSize, ...hiddenLayers, 2];
  const layers: LinearLayer[] = [];
  for (let i = 0; i < sizes.length - 1; i++) {
    layers.push({
      inputSize: sizes[i],
      outputSize: sizes[i + 1],
      weights: xavierInit(sizes[i], sizes[i + 1], rng),
      bias: new Float64Array(sizes[i + 1]),
    });
  }
  const paramCount = layers.reduce((sum, layer) => sum + layer.weights.length + layer.bias.length, 0);
  return {
    layers,
    activation,
    optimizer,
    learningRate,
    adamStep: 0,
    adamM: new Float64Array(paramCount),
    adamV: new Float64Array(paramCount),
  };
}

function activate(value: number, activation: ActivationId): number {
  switch (activation) {
    case "relu":
      return value > 0 ? value : 0;
    case "sigmoid":
      return 1 / (1 + Math.exp(-value));
    default:
      return Math.tanh(value);
  }
}

function activateDerivative(preActivation: number, activation: ActivationId): number {
  switch (activation) {
    case "relu":
      return preActivation > 0 ? 1 : 0;
    case "sigmoid": {
      const s = 1 / (1 + Math.exp(-preActivation));
      return s * (1 - s);
    }
    default: {
      const t = Math.tanh(preActivation);
      return 1 - t * t;
    }
  }
}

function linearForward(layer: LinearLayer, input: Float64Array): Float64Array {
  const out = new Float64Array(layer.outputSize);
  for (let o = 0; o < layer.outputSize; o++) {
    let sum = layer.bias[o];
    for (let i = 0; i < layer.inputSize; i++) {
      sum += layer.weights[o * layer.inputSize + i] * input[i];
    }
    out[o] = sum;
  }
  return out;
}

function forwardSample(mlp: MLP, input: number[]): { logits: number[]; cache: ForwardCache } {
  const cache: ForwardCache = { inputs: [], preActivations: [] };
  let current = Float64Array.from(input);

  for (let layerIndex = 0; layerIndex < mlp.layers.length; layerIndex++) {
    const layer = mlp.layers[layerIndex];
    cache.inputs.push(current);
    const preActivation = linearForward(layer, current);
    cache.preActivations.push(preActivation);
    if (layerIndex < mlp.layers.length - 1) {
      current = Float64Array.from(preActivation, (value) => activate(value, mlp.activation));
    } else {
      current = new Float64Array(preActivation);
    }
  }

  return { logits: Array.from(current), cache };
}

export function forward(mlp: MLP, batch: number[][]): { logits: number[][]; caches: ForwardCache[] } {
  const caches: ForwardCache[] = [];
  const logits = batch.map((row) => {
    const result = forwardSample(mlp, row);
    caches.push(result.cache);
    return result.logits;
  });
  return { logits, caches };
}

function softmaxRow(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function predictProbabilities(mlp: MLP, batch: number[][]): number[] {
  const { logits } = forward(mlp, batch);
  return logits.map((row) => softmaxRow(row)[1]);
}

/**
 * Accumulate SUM of per-example grads (not averaged) into a flat vector matching
 * {@link flattenWeights} layout. Does not update weights / Adam.
 */
export function accumulateBatchGradSums(
  mlp: MLP,
  batch: number[][],
  labels: number[],
): { grads: Float64Array; count: number; lossSum: number } {
  const grads = new Float64Array(
    mlp.layers.reduce((sum, layer) => sum + layer.weights.length + layer.bias.length, 0),
  );
  let lossSum = 0;
  const batchSize = batch.length;

  for (let sampleIndex = 0; sampleIndex < batchSize; sampleIndex++) {
    const { logits, cache } = forwardSample(mlp, batch[sampleIndex]!);
    const probs = softmaxRow(logits);
    const target = labels[sampleIndex]!;
    lossSum -= Math.log(Math.max(probs[target]!, 1e-12));

    let delta = Float64Array.from(probs);
    delta[target]! -= 1;

    // Write into a temp layer-grad structure then flatten (sum, not mean).
    const layerGrads = mlp.layers.map((layer) => ({
      weights: new Float64Array(layer.weights.length),
      bias: new Float64Array(layer.bias.length),
    }));

    for (let layerIndex = mlp.layers.length - 1; layerIndex >= 0; layerIndex--) {
      const layer = mlp.layers[layerIndex]!;
      const input = cache.inputs[layerIndex]!;
      const nextDelta = new Float64Array(layer.inputSize);

      for (let o = 0; o < layer.outputSize; o++) {
        const g = delta[o]!;
        layerGrads[layerIndex]!.bias[o]! += g;
        for (let i = 0; i < layer.inputSize; i++) {
          layerGrads[layerIndex]!.weights[o * layer.inputSize + i]! += g * input[i]!;
          nextDelta[i]! += layer.weights[o * layer.inputSize + i]! * g;
        }
      }

      if (layerIndex > 0) {
        const prevPreActivation = cache.preActivations[layerIndex - 1]!;
        delta = Float64Array.from(nextDelta, (value, i) =>
          value * activateDerivative(prevPreActivation[i]!, mlp.activation),
        );
      }
    }

    let offset = 0;
    for (const g of layerGrads) {
      for (let i = 0; i < g.weights.length; i++) grads[offset++]! += g.weights[i]!;
      for (let i = 0; i < g.bias.length; i++) grads[offset++]! += g.bias[i]!;
    }
  }

  return { grads, count: batchSize, lossSum };
}

/** Apply mean grads (flat, same layout as flattenWeights) via the MLP optimizer. */
export function applyFlatGradMeans(mlp: MLP, gradMeans: Float64Array): void {
  const layerGrads = mlp.layers.map((layer) => ({
    weights: new Float64Array(layer.weights.length),
    bias: new Float64Array(layer.bias.length),
  }));
  let offset = 0;
  for (const g of layerGrads) {
    for (let i = 0; i < g.weights.length; i++) g.weights[i] = gradMeans[offset++]!;
    for (let i = 0; i < g.bias.length; i++) g.bias[i] = gradMeans[offset++]!;
  }
  applyGradients(mlp, layerGrads);
}

export function trainBatch(mlp: MLP, batch: number[][], labels: number[]): number {
  const batchSize = batch.length;
  if (batchSize === 0) return 0;
  const { grads, lossSum } = accumulateBatchGradSums(mlp, batch, labels);
  // Convert sums → means for applyGradients (matches previous trainBatch behaviour).
  for (let i = 0; i < grads.length; i++) grads[i]! /= batchSize;
  applyFlatGradMeans(mlp, grads);
  return lossSum / batchSize;
}

function applyGradients(
  mlp: MLP,
  layerGrads: Array<{ weights: Float64Array; bias: Float64Array }>,
): void {
  // Keep flat adamM/adamV buffers for Adam + RMSProp (v in adamV; m unused for RMSProp).
  if (mlp.optimizer === "SGD") {
    for (let layerIndex = 0; layerIndex < mlp.layers.length; layerIndex++) {
      const layer = mlp.layers[layerIndex]!;
      const grad = layerGrads[layerIndex]!;
      for (let i = 0; i < layer.weights.length; i++) {
        layer.weights[i]! -= mlp.learningRate * grad.weights[i]!;
      }
      for (let i = 0; i < layer.bias.length; i++) {
        layer.bias[i]! -= mlp.learningRate * grad.bias[i]!;
      }
    }
    return;
  }

  mlp.adamStep += 1;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const rmsDecay = 0.9;
  const epsilon = 1e-8;
  const biasCorrection1 = 1 - beta1 ** mlp.adamStep;
  const biasCorrection2 = 1 - beta2 ** mlp.adamStep;
  let offset = 0;

  for (let layerIndex = 0; layerIndex < mlp.layers.length; layerIndex++) {
    const layer = mlp.layers[layerIndex]!;
    const grad = layerGrads[layerIndex]!;
    const blocks = [layer.weights, layer.bias] as const;
    const gradBlocks = [grad.weights, grad.bias] as const;
    for (let block = 0; block < 2; block++) {
      const array = blocks[block]!;
      const gradArray = gradBlocks[block]!;
      for (let i = 0; i < array.length; i++) {
        const g = gradArray[i]!;
        if (mlp.optimizer === "RMSProp") {
          mlp.adamV[offset] = rmsDecay * mlp.adamV[offset]! + (1 - rmsDecay) * g * g;
          array[i]! -= (mlp.learningRate * g) / (Math.sqrt(mlp.adamV[offset]!) + epsilon);
        } else {
          mlp.adamM[offset] = beta1 * mlp.adamM[offset]! + (1 - beta1) * g;
          mlp.adamV[offset] = beta2 * mlp.adamV[offset]! + (1 - beta2) * g * g;
          const mHat = mlp.adamM[offset]! / biasCorrection1;
          const vHat = mlp.adamV[offset]! / biasCorrection2;
          array[i]! -= (mlp.learningRate * mHat) / (Math.sqrt(vHat) + epsilon);
        }
        offset += 1;
      }
    }
  }
}

export function flattenWeights(mlp: MLP): Float64Array {
  const total = mlp.layers.reduce((sum, layer) => sum + layer.weights.length + layer.bias.length, 0);
  const vector = new Float64Array(total);
  let offset = 0;
  for (const layer of mlp.layers) {
    vector.set(layer.weights, offset);
    offset += layer.weights.length;
    vector.set(layer.bias, offset);
    offset += layer.bias.length;
  }
  return vector;
}

export function loadWeights(mlp: MLP, vector: Float64Array): void {
  let offset = 0;
  for (const layer of mlp.layers) {
    layer.weights.set(vector.subarray(offset, offset + layer.weights.length));
    offset += layer.weights.length;
    layer.bias.set(vector.subarray(offset, offset + layer.bias.length));
    offset += layer.bias.length;
  }
}

export function accuracy(mlp: MLP, batch: number[][], labels: number[]): number {
  let correct = 0;
  for (let i = 0; i < batch.length; i++) {
    const { logits } = forwardSample(mlp, batch[i]);
    const prediction = logits[0] >= logits[1] ? 0 : 1;
    if (prediction === labels[i]) correct += 1;
  }
  return correct / Math.max(batch.length, 1);
}

export function crossEntropyLoss(mlp: MLP, batch: number[][], labels: number[]): number {
  let loss = 0;
  for (let i = 0; i < batch.length; i++) {
    const { logits } = forwardSample(mlp, batch[i]);
    const probs = softmaxRow(logits);
    loss -= Math.log(Math.max(probs[labels[i]], 1e-12));
  }
  return loss / Math.max(batch.length, 1);
}

/** @deprecated use trainBatch */
export function computeLossAndGradients(mlp: MLP, batch: number[][], labels: number[]): number {
  return trainBatch(mlp, batch, labels);
}
