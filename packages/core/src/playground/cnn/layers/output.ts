import { cloneSignal, zeros1D, type Signal } from "../tensor";
import {
  applyBiasUpdate,
  applyRegularizedUpdate,
  type CnnRegularizationId,
} from "../regularization";
import { createOptState, resetOptState, type OptState, type PlaygroundOptimizerId } from "../../optimizers";
import { Layer, type LayerShape } from "./base";
import { Losses, type LossFunction } from "../loss";

/**
 * The terminal layer: a single sigmoid unit producing a probability, paired
 * with a {@link LossFunction}. The combined sigmoid+BCE backward collapses to
 * the clean `p − target` gradient, but to stay correct for any loss we keep the
 * two-step chain: dL/dp → dL/dz via the sigmoid derivative.
 */
export class OutputLayer extends Layer {
  /** Single output unit (binary classification). */
  units = 1;
  lossFn: LossFunction;
  /** Probability for the most recent forward pass (in (0,1)). */
  probability = 0.5;
  private weights: number[] = [];
  private bias = 0;
  private gradWeights: number[] = [];
  private gradBias = 0;
  private optWeights: OptState[] = [];
  private optBias: OptState = createOptState();
  private lastInput: number[] = [];
  private inUnits = 0;

  constructor(id: string, lossFn: LossFunction = Losses.BINARY_CROSS_ENTROPY) {
    super(id, "output", "1d");
    this.lossFn = lossFn;
  }

  label(): string {
    return "Output";
  }

  outputShape(input: LayerShape): LayerShape {
    if (input.kind !== "1d") throw new Error("Output expects a 1-D input");
    return { kind: "1d", channels: 1, length: 1 };
  }

  initParams(inUnits: number, rng: () => number): void {
    this.inUnits = inUnits;
    const bound = Math.sqrt(6 / Math.max(inUnits, 1));
    this.weights = new Array(inUnits);
    this.gradWeights = new Array(inUnits).fill(0);
    this.optWeights = new Array(inUnits);
    for (let i = 0; i < inUnits; i++) {
      this.weights[i] = (rng() * 2 - 1) * bound;
      this.optWeights[i] = createOptState();
    }
    this.bias = 0;
    this.gradBias = 0;
    this.optBias = createOptState();
  }

  forward(input: Signal): Signal {
    const x = input[0];
    if (this.weights.length === 0) this.initParams(x.length, Math.random);
    this.lastInput = x.slice();
    let acc = this.bias;
    for (let i = 0; i < x.length; i++) acc += this.weights[i] * x[i];
    this.probability = 1 / (1 + Math.exp(-acc));
    this.output = [[this.probability]];
    return this.output;
  }

  /**
   * `targetLabel` is the class label (∈ {0,1}) for the example currently being
   * back-propagated. The engine sets it via {@link setTarget} before calling
   * this method; absent a target the layer is treated as a plain sigmoid unit.
   */
  targetLabel = 0.5;

  setTarget(label: number): void {
    this.targetLabel = label;
  }

  backward(_gradOut: Signal): Signal {
    // dL/dp — from the loss; then dL/dz = dL/dp · sigmoid'(z) = dL/dp · p(1−p).
    const dLdp = this.lossFn.der(this.probability, this.targetLabel);
    const p = this.probability;
    const dZ = dLdp * p * (1 - p);

    this.gradBias += dZ;
    const gw = this.gradWeights;
    for (let i = 0; i < this.inUnits; i++) gw[i] += dZ * this.lastInput[i];

    const gIn = zeros1D(this.inUnits);
    for (let i = 0; i < this.inUnits; i++) gIn[i] += this.weights[i] * dZ;
    const wrapped: Signal = [gIn];
    this.inputGrad = wrapped;
    return wrapped;
  }

  /** Loss of the most recent forward pass against `target` (∈ {0,1}). */
  loss(target: number): number {
    return this.lossFn.error(this.probability, target);
  }

  updateParams(
    learningRate: number,
    regularization: CnnRegularizationId = "none",
    regularizationRate = 0,
    optimizer: PlaygroundOptimizerId = "SGD",
    optStep = 1,
  ): void {
    this.bias = applyBiasUpdate(
      this.bias,
      this.gradBias,
      learningRate,
      optimizer,
      this.optBias,
      optStep,
    );
    for (let i = 0; i < this.inUnits; i++) {
      this.weights[i] = applyRegularizedUpdate(
        this.weights[i]!,
        this.gradWeights[i]!,
        learningRate,
        regularization,
        regularizationRate,
        optimizer,
        this.optWeights[i]!,
        optStep,
      );
    }
  }

  zeroGrads(): void {
    this.gradBias = 0;
    this.gradWeights.fill(0);
  }

  clearOptimizerState(): void {
    for (const s of this.optWeights) resetOptState(s);
    resetOptState(this.optBias);
  }

  paramCount(): number {
    return (this.inUnits || 0) + 1;
  }

  reinitialize(rng: () => number): void {
    if (this.inUnits === 0) return;
    this.initParams(this.inUnits, rng);
  }

  weightMagnitude(): number {
    let sumSq = 0;
    for (let i = 0; i < this.inUnits; i++) sumSq += this.weights[i] ** 2;
    const n = this.inUnits || 1;
    return Math.tanh(Math.sqrt(sumSq / n));
  }

  snapshotWeights(): number[] {
    return this.weights.slice();
  }

  snapshotOutput(): Signal {
    return cloneSignal(this.output as Signal);
  }
}
