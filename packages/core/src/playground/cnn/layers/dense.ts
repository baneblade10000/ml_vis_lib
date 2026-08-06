import { cloneSignal, zeros1D, type Signal } from "../tensor";
import { activationById, type CnnActivationId } from "../activations";
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
    for (let o = 0; o < this.units; o++) {
      const w = new Array(inUnits);
      const gw = new Array(inUnits).fill(0);
      for (let i = 0; i < inUnits; i++) w[i] = (rng() * 2 - 1) * bound;
      this.weights[o] = w;
      this.gradWeights[o] = gw;
    }
    this.biases = new Array(this.units).fill(0);
    this.gradBiases = new Array(this.units).fill(0);
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

  updateParams(learningRate: number): void {
    for (let o = 0; o < this.units; o++) {
      this.biases[o] -= learningRate * this.gradBiases[o];
      const w = this.weights[o];
      const gw = this.gradWeights[o];
      for (let i = 0; i < this.inUnits; i++) w[i] -= learningRate * gw[i];
    }
  }

  zeroGrads(): void {
    for (let o = 0; o < this.units; o++) {
      this.gradBiases[o] = 0;
      this.gradWeights[o].fill(0);
    }
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
