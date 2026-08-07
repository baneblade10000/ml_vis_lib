import { cloneSignal, zeros1D, type Signal } from "../tensor";
import { activationById, type CnnActivationId } from "../activations";
import {
  applyBiasUpdate,
  applyRegularizedUpdate,
  type CnnRegularizationId,
} from "../regularization";
import { createOptState, resetOptState, type OptState, type PlaygroundOptimizerId } from "../../optimizers";
import { Layer, type LayerShape } from "./base";

/**
 * Fully-connected layer operating on a single-channel {@link Signal}
 * (the flattened representation). `W[o][i]` with bias `b[o]`; forward is a
 * plain matmul, backward the textbook dense backprop.
 */
export class DenseLayer extends Layer {
  units: number;
  activationId: CnnActivationId;

  /** `weights[out][in]`. */
  weights: number[][] = [];
  biases: number[] = [];
  private z: number[] = [];
  private lastInput: number[] = [];
  private gradWeights: number[][] = [];
  private gradBiases: number[] = [];
  private optWeights: OptState[][] = [];
  private optBiases: OptState[] = [];
  private inUnits = 0;

  constructor(id: string, units: number, activation: CnnActivationId = "linear") {
    super(id, "dense", "1d");
    this.units = units;
    this.activationId = activation;
  }

  label(): string {
    return `Dense ${this.units}`;
  }

  outputShape(input: LayerShape): LayerShape {
    if (input.kind !== "1d") throw new Error("Dense expects a 1-D input");
    return { kind: "1d", channels: 1, length: this.units };
  }

  initParams(inUnits: number, rng: () => number): void {
    this.inUnits = inUnits;
    const bound = Math.sqrt(6 / Math.max(inUnits, 1));
    this.weights = new Array(this.units);
    this.gradWeights = new Array(this.units);
    this.optWeights = new Array(this.units);
    for (let o = 0; o < this.units; o++) {
      const w = new Array(inUnits);
      const gw = new Array(inUnits).fill(0);
      const ow = new Array(inUnits);
      for (let i = 0; i < inUnits; i++) {
        w[i] = (rng() * 2 - 1) * bound;
        ow[i] = createOptState();
      }
      this.weights[o] = w;
      this.gradWeights[o] = gw;
      this.optWeights[o] = ow;
    }
    this.biases = new Array(this.units).fill(0);
    this.gradBiases = new Array(this.units).fill(0);
    this.optBiases = Array.from({ length: this.units }, () => createOptState());
  }

  forward(input: Signal): Signal {
    const x = input[0];
    if (this.weights.length === 0) {
      this.initParams(x.length, Math.random);
    }
    const fn = activationById(this.activationId);
    this.lastInput = x.slice();
    const z = new Array(this.units);
    const out = [new Array(this.units)];
    for (let o = 0; o < this.units; o++) {
      const w = this.weights[o];
      let acc = this.biases[o];
      for (let i = 0; i < x.length; i++) acc += w[i] * x[i];
      z[o] = acc;
      out[0][o] = fn.output(acc);
    }
    this.z = z;
    this.output = out;
    return out;
  }

  backward(gradOut: Signal): Signal {
    const fn = activationById(this.activationId);
    const gOut = gradOut[0];
    const dZ = new Array(this.units);
    for (let o = 0; o < this.units; o++) dZ[o] = gOut[o] * fn.der(this.z[o]);

    // Parameter grads — accumulate (zeroGrads() called once per batch).
    for (let o = 0; o < this.units; o++) {
      this.gradBiases[o] += dZ[o];
      const gw = this.gradWeights[o];
      const d = dZ[o];
      for (let i = 0; i < this.inUnits; i++) gw[i] += d * this.lastInput[i];
    }

    // Input grad: gIn[i] = Σ_o W[o][i] · dZ[o].
    const gIn = zeros1D(this.inUnits);
    for (let o = 0; o < this.units; o++) {
      const w = this.weights[o];
      const d = dZ[o];
      for (let i = 0; i < this.inUnits; i++) gIn[i] += w[i] * d;
    }
    const wrapped: Signal = [gIn];
    this.inputGrad = wrapped;
    return wrapped;
  }

  updateParams(
    learningRate: number,
    regularization: CnnRegularizationId = "none",
    regularizationRate = 0,
    optimizer: PlaygroundOptimizerId = "SGD",
    optStep = 1,
  ): void {
    for (let o = 0; o < this.units; o++) {
      this.biases[o] = applyBiasUpdate(
        this.biases[o]!,
        this.gradBiases[o]!,
        learningRate,
        optimizer,
        this.optBiases[o]!,
        optStep,
      );
      const w = this.weights[o]!;
      const gw = this.gradWeights[o]!;
      const ow = this.optWeights[o]!;
      for (let i = 0; i < this.inUnits; i++) {
        w[i] = applyRegularizedUpdate(
          w[i]!,
          gw[i]!,
          learningRate,
          regularization,
          regularizationRate,
          optimizer,
          ow[i]!,
          optStep,
        );
      }
    }
  }

  zeroGrads(): void {
    for (let o = 0; o < this.units; o++) {
      this.gradBiases[o] = 0;
      this.gradWeights[o].fill(0);
    }
  }

  override writeParams(dst: Float64Array, offset: number): number {
    let o = offset;
    for (let u = 0; u < this.units; u++) {
      for (let i = 0; i < this.inUnits; i++) dst[o++] = this.weights[u]![i]!;
    }
    for (let u = 0; u < this.units; u++) dst[o++] = this.biases[u]!;
    return o;
  }

  override readParams(src: Float64Array, offset: number): number {
    let o = offset;
    for (let u = 0; u < this.units; u++) {
      for (let i = 0; i < this.inUnits; i++) this.weights[u]![i] = src[o++]!;
    }
    for (let u = 0; u < this.units; u++) this.biases[u] = src[o++]!;
    return o;
  }

  override writeGrads(dst: Float64Array, offset: number): number {
    let o = offset;
    for (let u = 0; u < this.units; u++) {
      for (let i = 0; i < this.inUnits; i++) dst[o++] = this.gradWeights[u]![i]!;
    }
    for (let u = 0; u < this.units; u++) dst[o++] = this.gradBiases[u]!;
    return o;
  }

  override readGrads(src: Float64Array, offset: number): number {
    let o = offset;
    for (let u = 0; u < this.units; u++) {
      for (let i = 0; i < this.inUnits; i++) this.gradWeights[u]![i] = src[o++]!;
    }
    for (let u = 0; u < this.units; u++) this.gradBiases[u] = src[o++]!;
    return o;
  }

  clearOptimizerState(): void {
    for (const row of this.optWeights) for (const s of row) resetOptState(s);
    for (const s of this.optBiases) resetOptState(s);
  }

  paramCount(): number {
    return this.units * (this.inUnits || 0) + this.units;
  }

  reinitialize(rng: () => number): void {
    if (this.inUnits === 0) return;
    this.initParams(this.inUnits, rng);
  }

  weightMagnitude(): number {
    let sumSq = 0;
    let n = 0;
    for (let o = 0; o < this.units; o++) {
      for (let i = 0; i < this.inUnits; i++) {
        sumSq += this.weights[o][i] ** 2;
        n++;
      }
    }
    if (n === 0) return 0;
    return Math.tanh(Math.sqrt(sumSq / n));
  }

  setUnits(units: number, rng: () => number): void {
    if (this.inUnits === 0 || units === this.units) return;
    const prevWeights = this.weights.map((row) => row.slice());
    const prevBiases = this.biases.slice();
    this.units = units;
    this.initParams(this.inUnits, rng);
    const keepRows = Math.min(prevWeights.length, units);
    const keepCols = Math.min(this.inUnits, prevWeights[0]?.length ?? 0);
    for (let o = 0; o < keepRows; o++) {
      for (let i = 0; i < keepCols; i++) this.weights[o][i] = prevWeights[o][i];
      this.biases[o] = prevBiases[o];
    }
  }

  snapshotOutput(): Signal {
    return cloneSignal(this.output as Signal);
  }
}
