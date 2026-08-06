import {
  cloneKernel1D,
  zeros1D,
  zerosSignal,
  type Kernel1D,
  type Signal,
} from "../tensor";
import { activationById, type CnnActivationId } from "../activations";
import { Layer, type LayerShape } from "./base";

interface Conv1DConfig {
  filters: number;
  kernelSize: number;
  stride: number;
  padding: number;
  activation: CnnActivationId;
}

/**
 * 1-D convolution (cross-correlation) along the length axis.
 *
 * The forward/backward math is the 1-D reduction of {@link Conv2DLayer}:
 * `out[o][p] = bias[o] + Σ_i Σ_k W[o][i][k] · in[i][p·s − pad + k]`.
 */
export class Conv1DLayer extends Layer {
  filters: number;
  readonly kernelSize: number;
  readonly stride: number;
  readonly padding: number;
  activationId: CnnActivationId;

  /** `kernels[out][in][k]`. */
  kernels: Kernel1D;
  biases: number[];
  private z: Signal = [];
  private paddedInput: Signal = [];
  private gradKernels: Kernel1D;
  private gradBiases: number[];

  constructor(id: string, config: Conv1DConfig) {
    super(id, "conv1d", "1d");
    this.filters = config.filters;
    this.kernelSize = config.kernelSize;
    this.stride = config.stride;
    this.padding = config.padding;
    this.activationId = config.activation;
    this.kernels = [];
    this.biases = new Array(this.filters).fill(0);
    this.gradKernels = [];
    this.gradBiases = new Array(this.filters).fill(0);
  }

  label(): string {
    return `Conv1D ${this.kernelSize} · ${this.filters}`;
  }

  outputShape(input: LayerShape): LayerShape {
    if (input.kind !== "1d") throw new Error("Conv1D expects a 1-D input");
    // Clamp at 1 so a too-large kernel never yields a degenerate length.
    const length = Math.max(1, Math.floor((input.length + 2 * this.padding - this.kernelSize) / this.stride) + 1);
    return { kind: "1d", channels: this.filters, length };
  }

  initParams(inChannels: number, rng: () => number): void {
    const fanIn = inChannels * this.kernelSize;
    const bound = Math.sqrt(6 / Math.max(fanIn, 1));
    this.kernels = new Array(this.filters);
    this.gradKernels = new Array(this.filters);
    for (let o = 0; o < this.filters; o++) {
      this.kernels[o] = new Array(inChannels);
      this.gradKernels[o] = new Array(inChannels);
      for (let i = 0; i < inChannels; i++) {
        const w = new Array(this.kernelSize);
        const gw = new Array(this.kernelSize).fill(0);
        for (let k = 0; k < this.kernelSize; k++) w[k] = (rng() * 2 - 1) * bound;
        this.kernels[o][i] = w;
        this.gradKernels[o][i] = gw;
      }
    }
    this.biases = new Array(this.filters).fill(0);
    this.gradBiases = new Array(this.filters).fill(0);
  }

  private zeroGradsInternal(): void {
    for (let o = 0; o < this.filters; o++) {
      for (let i = 0; i < this.gradKernels[o].length; i++) {
        this.gradKernels[o][i].fill(0);
      }
    }
    this.gradBiases.fill(0);
  }

  zeroGrads(): void {
    this.zeroGradsInternal();
  }

  forward(input: Signal): Signal {
    const inChannels = input.length;
    const inLen = input[0].length;
    if (this.kernels.length === 0) {
      this.initParams(inChannels, Math.random);
    }
    const outShape = this.outputShape({ kind: "1d", channels: inChannels, length: inLen });
    const outLen = outShape.kind === "1d" ? outShape.length : 0;
    const padded = this.padding > 0 ? this.pad(input, this.padding) : input;
    this.paddedInput = padded;
    const pLen = padded[0].length;

    const fn = activationById(this.activationId);
    const z = zerosSignal(this.filters, outLen);
    const out = zerosSignal(this.filters, outLen);
    const s = this.stride;
    const k = this.kernelSize;

    for (let o = 0; o < this.filters; o++) {
      const wBank = this.kernels[o];
      const b = this.biases[o];
      for (let p = 0; p < outLen; p++) {
        let acc = b;
        const base = p * s;
        for (let i = 0; i < inChannels; i++) {
          const w = wBank[i];
          const inCh = padded[i];
          for (let kk = 0; kk < k; kk++) {
            const pos = base + kk;
            if (pos >= pLen) continue;
            acc += w[kk] * inCh[pos];
          }
        }
        z[o][p] = acc;
        out[o][p] = fn.output(acc);
      }
    }
    this.z = z;
    this.output = out;
    return out;
  }

  backward(gradOut: Signal): Signal {
    const inChannels = this.paddedInput.length;
    const pLen = this.paddedInput[0].length;
    const outLen = this.output.length ? this.output[0].length : 0;
    const fn = activationById(this.activationId);
    const s = this.stride;
    const k = this.kernelSize;

    const dZ = zerosSignal(this.filters, outLen);
    for (let o = 0; o < this.filters; o++) {
      for (let p = 0; p < outLen; p++) {
        dZ[o][p] = gradOut[o][p] * fn.der(this.z[o][p]);
      }
    }

    // Accumulate grads (zeroGrads() is called once per batch).
    for (let o = 0; o < this.filters; o++) {
      let gb = 0;
      for (let p = 0; p < outLen; p++) {
        const g = dZ[o][p];
        gb += g;
        const base = p * s;
        for (let i = 0; i < inChannels; i++) {
          const gw = this.gradKernels[o][i];
          const inCh = this.paddedInput[i];
          for (let kk = 0; kk < k; kk++) {
            const pos = base + kk;
            if (pos >= pLen) continue;
            gw[kk] += g * inCh[pos];
          }
        }
      }
      this.gradBiases[o] += gb;
    }

    const gradPadded = zerosSignal(inChannels, pLen);
    for (let i = 0; i < inChannels; i++) {
      for (let o = 0; o < this.filters; o++) {
        const w = this.kernels[o][i];
        for (let p = 0; p < outLen; p++) {
          const g = dZ[o][p];
          const base = p * s;
          for (let kk = 0; kk < k; kk++) {
            const pos = base + kk;
            if (pos >= pLen) continue;
            // No flip: forward is cross-correlation, so the transpose scatters
            // gradOut back to its input cell with the same kernel index.
            gradPadded[i][pos] += g * w[kk];
          }
        }
      }
    }
    this.inputGrad = this.padding > 0 ? this.unpad(gradPadded, this.padding) : gradPadded;
    return this.inputGrad;
  }

  updateParams(learningRate: number): void {
    for (let o = 0; o < this.filters; o++) {
      this.biases[o] -= learningRate * this.gradBiases[o];
      for (let i = 0; i < this.kernels[o].length; i++) {
        const w = this.kernels[o][i];
        const gw = this.gradKernels[o][i];
        for (let kk = 0; kk < this.kernelSize; kk++) w[kk] -= learningRate * gw[kk];
      }
    }
  }

  paramCount(): number {
    if (this.kernels.length === 0) return 0;
    return this.filters * this.kernels[0].length * this.kernelSize + this.filters;
  }

  reinitialize(rng: () => number): void {
    if (this.kernels.length === 0) return;
    this.initParams(this.kernels[0].length, rng);
  }

  weightMagnitude(): number {
    let sumSq = 0;
    let n = 0;
    for (let o = 0; o < this.filters; o++) {
      for (let i = 0; i < this.kernels[o].length; i++) {
        for (let kk = 0; kk < this.kernelSize; kk++) {
          sumSq += this.kernels[o][i][kk] ** 2;
          n++;
        }
      }
    }
    if (n === 0) return 0;
    return Math.tanh(Math.sqrt(sumSq / n));
  }

  snapshotKernels(): Kernel1D {
    return cloneKernel1D(this.kernels);
  }

  /** One display vector per filter: the sum of input-channel kernels. */
  featureKernels(): number[][] {
    const out: number[][] = [];
    for (let o = 0; o < this.filters; o++) {
      const acc = zeros1D(this.kernelSize);
      for (let i = 0; i < this.kernels[o].length; i++) {
        for (let kk = 0; kk < this.kernelSize; kk++) acc[kk] += this.kernels[o][i][kk];
      }
      out.push(acc);
    }
    return out;
  }

  setFilters(filters: number, rng: () => number): void {
    if (this.kernels.length === 0 || filters === this.filters) return;
    const inChannels = this.kernels[0].length;
    const prevKernels = cloneKernel1D(this.kernels);
    const prevBiases = this.biases.slice();
    this.filters = filters;
    this.initParams(inChannels, rng);
    const keep = Math.min(prevKernels.length, filters);
    for (let o = 0; o < keep; o++) {
      for (let i = 0; i < inChannels; i++) {
        this.kernels[o][i] = prevKernels[o][i];
      }
      this.biases[o] = prevBiases[o];
    }
  }

  private pad(input: Signal, pad: number): Signal {
    if (pad === 0) return input.map((row) => row.slice());
    const channels = input.length;
    const out = new Array(channels);
    for (let i = 0; i < channels; i++) {
      out[i] = new Array(pad).fill(0).concat(input[i]).concat(new Array(pad).fill(0));
    }
    return out;
  }

  private unpad(padded: Signal, pad: number): Signal {
    if (pad === 0) return padded;
    return padded.map((row) => row.slice(pad, row.length - pad));
  }
}
