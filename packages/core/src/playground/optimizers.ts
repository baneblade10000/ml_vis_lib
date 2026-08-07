/**
 * Shared first-order optimizers for the interactive TF + CNN playgrounds.
 * Hyperparameters match the classic defaults used by the offline `mlp.ts` Adam path.
 */

export type PlaygroundOptimizerId = "SGD" | "RMSProp" | "Adam";

export const PLAYGROUND_OPTIMIZERS: readonly PlaygroundOptimizerId[] = [
  "SGD",
  "RMSProp",
  "Adam",
];

const BETA1 = 0.9;
const BETA2 = 0.999;
const RMS_DECAY = 0.9;
const EPSILON = 1e-8;

/** Per-parameter moment buffers (Adam: m+v, RMSProp: v only). */
export interface OptState {
  m: number;
  v: number;
}

export function createOptState(): OptState {
  return { m: 0, v: 0 };
}

export function resetOptState(state: OptState): void {
  state.m = 0;
  state.v = 0;
}

/**
 * Compute the parameter delta to subtract: `param -= delta`.
 * `step` is 1-based and only used by Adam bias correction.
 */
export function optimizerDelta(
  grad: number,
  state: OptState,
  optimizer: PlaygroundOptimizerId,
  learningRate: number,
  step: number,
): number {
  switch (optimizer) {
    case "SGD":
      return learningRate * grad;
    case "RMSProp": {
      state.v = RMS_DECAY * state.v + (1 - RMS_DECAY) * grad * grad;
      return (learningRate * grad) / (Math.sqrt(state.v) + EPSILON);
    }
    case "Adam": {
      const t = Math.max(1, step);
      state.m = BETA1 * state.m + (1 - BETA1) * grad;
      state.v = BETA2 * state.v + (1 - BETA2) * grad * grad;
      const mHat = state.m / (1 - BETA1 ** t);
      const vHat = state.v / (1 - BETA2 ** t);
      return (learningRate * mHat) / (Math.sqrt(vHat) + EPSILON);
    }
  }
}
