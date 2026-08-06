import { cloneSignal, zerosSignal, type Signal } from "../tensor";
import { Layer, type LayerShape } from "./base";
import type { PoolKind2D } from "./pool2d";

/**
 * 1-D length-2 stride-2 pooling. The 1-D reduction of {@link Pool2DLayer}:
 * max-pool routes the gradient to the winning index; avg-pool splits it evenly.
 */
export class Pool1DLayer extends Layer {
  readonly poolSize: number;
  readonly poolKind: PoolKind2D;
  private argmax: number[][] = [];
  private inLen = 0;

  constructor(id: string, poolKind: PoolKind2D = "max", poolSize = 2) {
    super(id, "pool1d", "1d");
    this.poolKind = poolKind;
    this.poolSize = poolSize;
  }

  label(): string {
    return `${this.poolKind === "max" ? "Max" : "Avg"} Pool ${this.poolSize}`;
  }

  outputShape(input: LayerShape): LayerShape {
    if (input.kind !== "1d") throw new Error("Pool1D expects a 1-D input");
    return { kind: "1d", channels: input.channels, length: Math.floor(input.length / this.poolSize) };
  }

  forward(input: Signal): Signal {
    const channels = input.length;
    this.inLen = input[0].length;
    const outLen = Math.floor(this.inLen / this.poolSize);
    const out = zerosSignal(channels, outLen);
    this.argmax = new Array(channels);
    for (let c = 0; c < channels; c++) {
      this.argmax[c] = new Array(outLen);
      for (let p = 0; p < outLen; p++) {
        const base = p * this.poolSize;
        if (this.poolKind === "max") {
          let best = -Infinity;
          let bestIdx = base;
          for (let kk = 0; kk < this.poolSize; kk++) {
            const v = input[c][base + kk];
            if (v > best) {
              best = v;
              bestIdx = base + kk;
            }
          }
          out[c][p] = best;
          this.argmax[c][p] = bestIdx;
        } else {
          let sum = 0;
          for (let kk = 0; kk < this.poolSize; kk++) sum += input[c][base + kk];
          out[c][p] = sum / this.poolSize;
          this.argmax[c][p] = base;
        }
      }
    }
    this.output = out;
    return out;
  }

  backward(gradOut: Signal): Signal {
    const channels = gradOut.length;
    const gradIn = zerosSignal(channels, this.inLen);
    const outLen = gradOut[0].length;
    const denom = this.poolSize;
    for (let c = 0; c < channels; c++) {
      for (let p = 0; p < outLen; p++) {
        const g = gradOut[c][p];
        if (this.poolKind === "max") {
          gradIn[c][this.argmax[c][p]] += g;
        } else {
          const each = g / denom;
          const base = p * this.poolSize;
          for (let kk = 0; kk < this.poolSize; kk++) gradIn[c][base + kk] += each;
        }
      }
    }
    this.inputGrad = gradIn;
    return gradIn;
  }

  updateParams(): void {
    /* no params */
  }

  zeroGrads(): void {
    /* no params */
  }

  paramCount(): number {
    return 0;
  }

  reinitialize(): void {
    /* no params */
  }

  weightMagnitude(): number | null {
    return null;
  }

  snapshotOutput(): Signal {
    return cloneSignal(this.output as Signal);
  }
}
