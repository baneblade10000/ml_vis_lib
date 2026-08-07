import {
  createOptState,
  optimizerDelta,
  type OptState,
  type PlaygroundOptimizerId,
} from "../optimizers";

/** Same ids as the dense playground so the UI can share option lists. */
export type CnnRegularizationId = "none" | "L1" | "L2";

export const CNN_REGULARIZATIONS: readonly CnnRegularizationId[] = ["none", "L1", "L2"];

export const CNN_REGULARIZATION_RATES = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10] as const;

/** Derivative of the regularization penalty w.r.t. a single weight. */
export function regularizationDerivative(weight: number, kind: CnnRegularizationId): number {
  switch (kind) {
    case "L1":
      return weight < 0 ? -1 : weight > 0 ? 1 : 0;
    case "L2":
      return weight;
    default:
      return 0;
  }
}

export function ensureOptState(state: OptState | undefined): OptState {
  return state ?? createOptState();
}

/**
 * One optimizer step on a weight with optional L1/L2.
 * L1 zeros the weight when the update would cross zero (same soft-threshold as MLP).
 */
export function applyRegularizedUpdate(
  weight: number,
  grad: number,
  learningRate: number,
  regularization: CnnRegularizationId,
  regularizationRate: number,
  optimizer: PlaygroundOptimizerId = "SGD",
  optState: OptState = createOptState(),
  optStep = 1,
): number {
  const regulDer = regularizationDerivative(weight, regularization);
  const effectiveGrad = grad + regularizationRate * regulDer;
  const next = weight - optimizerDelta(effectiveGrad, optState, optimizer, learningRate, optStep);
  if (regularization === "L1" && weight * next < 0) return 0;
  return next;
}

/** Bias update (never regularized). */
export function applyBiasUpdate(
  bias: number,
  grad: number,
  learningRate: number,
  optimizer: PlaygroundOptimizerId = "SGD",
  optState: OptState = createOptState(),
  optStep = 1,
): number {
  return bias - optimizerDelta(grad, optState, optimizer, learningRate, optStep);
}
