import type { AutogradOp } from "./types";

/**
 * Evaluate an operation given its ordered input values.
 * Leaf ops (input / const) never reach here — their value is stored directly.
 */
export function evaluateOp(op: AutogradOp, inputs: number[]): number {
  switch (op) {
    case "add":
      return inputs.reduce((a, b) => a + b, 0);
    case "mul":
      return inputs.reduce((a, b) => a * b, 1);
    case "sub":
      return (inputs[0] ?? 0) - (inputs[1] ?? 0);
    case "div":
      return (inputs[0] ?? 0) / (inputs[1] ?? 1);
    case "neg":
      return -(inputs[0] ?? 0);
    case "exp":
      return Math.exp(inputs[0] ?? 0);
    case "tanh":
      return Math.tanh(inputs[0] ?? 0);
    case "relu":
      return Math.max(0, inputs[0] ?? 0);
    default:
      return inputs[0] ?? 0;
  }
}

/**
 * Local partial derivatives d(out)/d(inputᵢ) for each input, given the input
 * values and the already-computed output. Aligned with the `inputs` order.
 */
export function localDerivatives(op: AutogradOp, inputs: number[], out: number): number[] {
  switch (op) {
    case "add":
      return inputs.map(() => 1);
    case "mul":
      // d/dxᵢ (∏ xⱼ) = ∏_{j≠i} xⱼ = product / xᵢ (guard against zero factors).
      return inputs.map((_, i) => {
        let prod = 1;
        for (let j = 0; j < inputs.length; j++) {
          if (j !== i) prod *= inputs[j];
        }
        return prod;
      });
    case "sub":
      return [1, -1];
    case "div": {
      const a = inputs[0] ?? 0;
      const b = inputs[1] ?? 1;
      return [1 / b, -a / (b * b)];
    }
    case "neg":
      return [-1];
    case "exp":
      return [out];
    case "tanh":
      return [1 - out * out];
    case "relu":
      return [(inputs[0] ?? 0) > 0 ? 1 : 0];
    default:
      return inputs.map(() => 0);
  }
}
