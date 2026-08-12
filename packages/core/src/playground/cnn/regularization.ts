/** Regularization ids / rates for the CNN playground UI / WASM config. */

export type CnnRegularizationId = "none" | "L1" | "L2";

export const CNN_REGULARIZATIONS: readonly CnnRegularizationId[] = ["none", "L1", "L2"];

export const CNN_REGULARIZATION_RATES = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10] as const;
