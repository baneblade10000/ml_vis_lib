/**
 * Loss functions for binary classification (label ∈ {0, 1}).
 *
 * The output layer is a single sigmoid unit, so binary cross-entropy is the
 * natural training objective. Squared error is kept as a fallback option for
 * parity with the feedforward playground.
 */

export interface LossFunction {
  /** Average loss of `output` (a probability in (0,1)) against `target` (∈{0,1}). */
  error: (output: number, target: number) => number;
  /** dLoss/dOutput — the gradient that starts backpropagation. */
  der: (output: number, target: number) => number;
}

export class Losses {
  /**
   * Binary cross-entropy with a small clamp so logs stay finite even for a
   * saturated sigmoid. Combined with the sigmoid output this yields the clean
   * gradient `output − target` (see `OutputLayer.backward`).
   */
  public static BINARY_CROSS_ENTROPY: LossFunction = {
    error: (output, target) => {
      const p = clampProb(output);
      const q = 1 - p;
      return -(target * Math.log(p) + (1 - target) * Math.log(q));
    },
    der: (output, target) => {
      const p = clampProb(output);
      return (p - target) / (p * (1 - p));
    },
  };

  public static SQUARED: LossFunction = {
    error: (output, target) => 0.5 * (output - target) ** 2,
    der: (output, target) => output - target,
  };
}

function clampProb(p: number): number {
  const eps = 1e-7;
  if (p < eps) return eps;
  if (p > 1 - eps) return 1 - eps;
  return p;
}
