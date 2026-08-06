import {
  cloneKernel2D,
  cloneVolume,
  zeros2D,
  zerosVolume,
  type Kernel2D,
  type Map2D,
  type Volume,
} from "../tensor";
import { activationById, type CnnActivationId } from "../activations";
import { Layer, type LayerShape } from "./base";

interface Conv2DConfig {
  /** Number of output channels (filters). */
  filters: number;
  /** Square kernel size (e.g. 3 → 3×3). */
  kernelSize: number;
  /** Stride in both axes. */
  stride: number;
  /** Zero-padding on each side. */
  padding: number;
  activation: CnnActivationId;
}

/**
 * 2-D convolution layer (strictly: cross-correlation, matching the deep-learning
 * convention used by every major framework).
 *
 * Forward: `out[o][or][oc] = bias[o] + Σ_{i} Σ_{kr,kc} W[o][i][kr][kc] · in[i][or·s − p + kr][oc·s − p + kc]`
 *
 * Backprop accumulates:
 *  - `gradW[o][i][kr][kc] += gradOut[o][or][oc] · in[i][or·s − p + kr][oc·s − p + kc]`
 *  - `gradB[o]            += Σ gradOut[o][or][oc]`
 *  - `gradIn[i][…]` via full-padding reverse pass.
 */
export class Conv2DLayer extends Layer {
  filters: number;
  readonly kernelSize: number;
  readonly stride: number;
  readonly padding: number;
  activationId: CnnActivationId;

  /** `kernels[out][in][kr][kc]`. */
  kernels: Kernel2D;
  biases: number[];
  /** Pre-activation totals retained from the last forward pass (for the activation derivative). */
  private z: Volume = [];
  /** Cached padded input (input to the weight gradient). */
  private paddedInput: Volume = [];

  private gradKernels: Kernel2D;
  private gradBiases: number[];

  constructor(id: string, config: Conv2DConfig) {
    super(id, "conv2d", "2d");
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
    return `Conv2D ${this.kernelSize}×${this.kernelSize} · ${this.filters}`;
  }

  outputShape(input: LayerShape): LayerShape {
    if (input.kind !== "2d") throw new Error("Conv2D expects a 2-D input");
    // Clamp at 1 so a too-large kernel never yields a degenerate (zero/negative)
    // spatial size — the interactive editor must not crash on odd architectures.
    const rows = Math.max(1, Math.floor((input.rows + 2 * this.padding - this.kernelSize) / this.stride) + 1);
    const cols = Math.max(1, Math.floor((input.cols + 2 * this.padding - this.kernelSize) / this.stride) + 1);
    return { kind: "2d", channels: this.filters, rows, cols };
  }

  /** Allocate kernels/biases once the upstream channel count is known. */
  initParams(inChannels: number, rng: () => number): void {
    const fanIn = inChannels * this.kernelSize * this.kernelSize;
    const bound = Math.sqrt(6 / Math.max(fanIn, 1));
    this.kernels = new Array(this.filters);
    this.gradKernels = new Array(this.filters);
    for (let o = 0; o < this.filters; o++) {
      this.kernels[o] = new Array(inChannels);
      this.gradKernels[o] = new Array(inChannels);
      for (let i = 0; i < inChannels; i++) {
        const w = new Array(this.kernelSize);
        const gw = new Array(this.kernelSize);
        for (let kr = 0; kr < this.kernelSize; kr++) {
          w[kr] = new Array(this.kernelSize);
          gw[kr] = new Array(this.kernelSize).fill(0);
          for (let kc = 0; kc < this.kernelSize; kc++) {
            w[kr][kc] = (rng() * 2 - 1) * bound;
          }
        }
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
        for (let kr = 0; kr < this.kernelSize; kr++) {
          this.gradKernels[o][i][kr].fill(0);
        }
      }
    }
    this.gradBiases.fill(0);
  }

  zeroGrads(): void {
    this.zeroGradsInternal();
  }

  forward(input: Volume): Volume {
    const inChannels = input.length;
    const inRows = input[0].length;
    const inCols = input[0][0].length;
    if (this.kernels.length === 0) {
      // Lazy init the first time we see real data.
      this.initParams(inChannels, Math.random);
    }

    const outShape = this.outputShape({ kind: "2d", channels: inChannels, rows: inRows, cols: inCols });
    const outRows = outShape.kind === "2d" ? outShape.rows : 0;
    const outCols = outShape.kind === "2d" ? outShape.cols : 0;

    const padded = this.padding > 0 ? this.pad(input, this.padding) : input;
    this.paddedInput = padded;

    const fn = activationById(this.activationId);
    const z = zerosVolume(this.filters, outRows, outCols);
    const out = zerosVolume(this.filters, outRows, outCols);
    const s = this.stride;
    const k = this.kernelSize;
    const pRows = padded[0].length;
    const pCols = padded[0][0].length;

    for (let o = 0; o < this.filters; o++) {
      const wBank = this.kernels[o];
      const b = this.biases[o];
      for (let or = 0; or < outRows; or++) {
        for (let oc = 0; oc < outCols; oc++) {
          let acc = b;
          const baseR = or * s;
          const baseC = oc * s;
          for (let i = 0; i < inChannels; i++) {
            const w = wBank[i];
            const inCh = padded[i];
            for (let kr = 0; kr < k; kr++) {
              const r = baseR + kr;
              if (r >= pRows) continue;
              const inRow = inCh[r];
              const wRow = w[kr];
              for (let kc = 0; kc < k; kc++) {
                const c = baseC + kc;
                if (c >= pCols) continue;
                acc += wRow[kc] * inRow[c];
              }
            }
          }
          z[o][or][oc] = acc;
          out[o][or][oc] = fn.output(acc);
        }
      }
    }
    this.z = z;
    this.output = out;
    return out;
  }

  backward(gradOut: Volume): Volume {
    const inChannels = this.paddedInput.length;
    const pRows = this.paddedInput[0].length;
    const pCols = this.paddedInput[0][0].length;
    const out = this.output as Volume;
    const outRows = out.length ? out[0].length : 0;
    const outCols = outRows ? out[0][0].length : 0;
    const fn = activationById(this.activationId);
    const s = this.stride;
    const k = this.kernelSize;

    // dZ = gradOut ⊙ activation'(z)
    const dZ = zerosVolume(this.filters, outRows, outCols);
    for (let o = 0; o < this.filters; o++) {
      for (let r = 0; r < outRows; r++) {
        for (let c = 0; c < outCols; c++) {
          dZ[o][r][c] = gradOut[o][r][c] * fn.der(this.z[o][r][c]);
        }
      }
    }

    // gradW and gradB — accumulate (zeroGrads() was called once per batch).
    for (let o = 0; o < this.filters; o++) {
      const wBank = this.gradKernels[o];
      let gb = 0;
      for (let r = 0; r < outRows; r++) {
        for (let c = 0; c < outCols; c++) {
          const g = dZ[o][r][c];
          gb += g;
          const baseR = r * s;
          const baseC = c * s;
          for (let i = 0; i < inChannels; i++) {
            const gw = wBank[i];
            const inCh = this.paddedInput[i];
            for (let kr = 0; kr < k; kr++) {
              const rr = baseR + kr;
              if (rr >= pRows) continue;
              const inRow = inCh[rr];
              const gwRow = gw[kr];
              for (let kc = 0; kc < k; kc++) {
                const cc = baseC + kc;
                if (cc >= pCols) continue;
                gwRow[kc] += g * inRow[cc];
              }
            }
          }
        }
      }
      this.gradBiases[o] += gb;
    }

    // gradIn (full convolution of dZ with the flipped kernel)
    const gradPadded = zerosVolume(inChannels, pRows, pCols);
    for (let i = 0; i < inChannels; i++) {
      for (let o = 0; o < this.filters; o++) {
        const wBank = this.kernels[o];
        const w = wBank[i];
        for (let r = 0; r < outRows; r++) {
          for (let c = 0; c < outCols; c++) {
            const g = dZ[o][r][c];
            const baseR = r * s;
            const baseC = c * s;
            for (let kr = 0; kr < k; kr++) {
              const rr = baseR + kr;
              if (rr >= pRows) continue;
              const gRow = gradPadded[i][rr];
              for (let kc = 0; kc < k; kc++) {
                const cc = baseC + kc;
                if (cc >= pCols) continue;
                // No flip: forward is cross-correlation, so the transpose uses the
                // same kernel indexing, scattering gradOut back to its input cell.
                gRow[cc] += g * w[kr][kc];
              }
            }
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
        for (let kr = 0; kr < this.kernelSize; kr++) {
          for (let kc = 0; kc < this.kernelSize; kc++) {
            w[kr][kc] -= learningRate * gw[kr][kc];
          }
        }
      }
    }
  }

  paramCount(): number {
    if (this.kernels.length === 0) return 0;
    return this.filters * this.kernels[0].length * this.kernelSize * this.kernelSize + this.filters;
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
        for (let kr = 0; kr < this.kernelSize; kr++) {
          for (let kc = 0; kc < this.kernelSize; kc++) {
            sumSq += this.kernels[o][i][kr][kc] ** 2;
            n++;
          }
        }
      }
    }
    if (n === 0) return 0;
    return Math.tanh(Math.sqrt(sumSq / n));
  }

  snapshotKernels(): Kernel2D {
    return cloneKernel2D(this.kernels);
  }

  /** Return this layer's kernels reshaped for display (one map per filter: sum of channels). */
  featureKernels(): Map2D[] {
    const out: Map2D[] = [];
    for (let o = 0; o < this.filters; o++) {
      const acc = zeros2D(this.kernelSize, this.kernelSize);
      for (let i = 0; i < this.kernels[o].length; i++) {
        for (let kr = 0; kr < this.kernelSize; kr++) {
          for (let kc = 0; kc < this.kernelSize; kc++) {
            acc[kr][kc] += this.kernels[o][i][kr][kc];
          }
        }
      }
      out.push(acc);
    }
    return out;
  }

  /**
   * Change the number of filters, preserving previously-learned weights where
   * the old and new filter counts overlap. New filters are randomly initialized.
   */
  setFilters(filters: number, rng: () => number): void {
    if (this.kernels.length === 0 || filters === this.filters) return;
    const inChannels = this.kernels[0].length;
    const prevKernels = cloneKernel2D(this.kernels);
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

  private pad(input: Volume, pad: number): Volume {
    if (pad === 0) return cloneVolume(input);
    const inChannels = input.length;
    const inRows = input[0].length;
    const inCols = input[0][0].length;
    const outRows = inRows + 2 * pad;
    const outCols = inCols + 2 * pad;
    const out = zerosVolume(inChannels, outRows, outCols);
    for (let i = 0; i < inChannels; i++) {
      for (let r = 0; r < inRows; r++) {
        for (let c = 0; c < inCols; c++) {
          out[i][r + pad][c + pad] = input[i][r][c];
        }
      }
    }
    return out;
  }

  private unpad(padded: Volume, pad: number): Volume {
    if (pad === 0) return padded;
    const inChannels = padded.length;
    const pRows = padded[0].length;
    const pCols = padded[0][0].length;
    const outRows = pRows - 2 * pad;
    const outCols = pCols - 2 * pad;
    const out = zerosVolume(inChannels, outRows, outCols);
    for (let i = 0; i < inChannels; i++) {
      for (let r = 0; r < outRows; r++) {
        for (let c = 0; c < outCols; c++) {
          out[i][r][c] = padded[i][r + pad][c + pad];
        }
      }
    }
    return out;
  }
}
