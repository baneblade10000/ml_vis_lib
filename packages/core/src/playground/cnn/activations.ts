import { Activations, type ActivationFunction } from "../network/nn";

/**
 * Activation registry for the convolutional-network engine.
 *
 * These reuse the proven activation implementations from the feedforward engine
 * (see `../network/nn.ts`) rather than duplicating their numerics, and add the
 * human-readable id used by the playground UI.
 */
export type CnnActivationId = "relu" | "tanh" | "sigmoid" | "linear";

export const CNN_ACTIVATIONS: Record<CnnActivationId, ActivationFunction> = {
  relu: Activations.RELU,
  tanh: Activations.TANH,
  sigmoid: Activations.SIGMOID,
  linear: Activations.LINEAR,
};

export const CNN_ACTIVATION_IDS = Object.keys(CNN_ACTIVATIONS) as CnnActivationId[];

export function activationById(id: CnnActivationId): ActivationFunction {
  const fn = CNN_ACTIVATIONS[id];
  if (!fn) throw new Error(`Unknown activation: ${id}`);
  return fn;
}

export { Activations };
export type { ActivationFunction };
