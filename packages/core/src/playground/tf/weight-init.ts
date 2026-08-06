/** Weight initialization schemes for the neural-network playground. */

export type WeightInitId = "uniform" | "xavier" | "he" | "normal" | "zeros";

export const WEIGHT_INITS: readonly WeightInitId[] = [
  "uniform",
  "xavier",
  "he",
  "normal",
  "zeros",
] as const;

/** Sample a single weight given fan-in / fan-out of the destination unit. */
export function sampleWeight(init: WeightInitId, fanIn: number, fanOut: number): number {
  const inSize = Math.max(fanIn, 1);
  const outSize = Math.max(fanOut, 1);
  switch (init) {
    case "zeros":
      return 0;
    case "xavier": {
      const limit = Math.sqrt(6 / (inSize + outSize));
      return (Math.random() * 2 - 1) * limit;
    }
    case "he": {
      const limit = Math.sqrt(6 / inSize);
      return (Math.random() * 2 - 1) * limit;
    }
    case "normal": {
      // Box–Muller; σ = 0.1 (small Gaussian, easy to visualize).
      const u = Math.max(Math.random(), 1e-12);
      const v = Math.random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * 0.1;
    }
    case "uniform":
    default:
      // TensorFlow Playground default: U[-0.5, 0.5).
      return Math.random() - 0.5;
  }
}

/** Bias for a non-input node under the selected scheme. */
export function sampleBias(init: WeightInitId, kind: string): number {
  if (kind === "sum") return 0;
  // Match classic TF Playground: small positive bias for uniform init.
  if (init === "uniform") return 0.1;
  return 0;
}
