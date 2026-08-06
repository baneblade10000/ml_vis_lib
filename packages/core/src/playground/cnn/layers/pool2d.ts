import { cloneVolume, zerosVolume, type Volume } from "../tensor";
import { Layer, type LayerShape } from "./base";

export type PoolKind2D = "max" | "avg";

/**
 * 2×2 pooling with stride 2. No learnable params — forward remembers the argmax
 * positions (for `max`) so the gradient can be routed to exactly those cells.
 */
export class Pool2DLayer extends Layer {
  readonly poolSize: number;
  readonly poolKind: PoolKind2D;
  /** For max-pooling: the input position (per channel) that won each output cell. */
  private argmax: { r: number; c: number }[][][] = [];
  private inRows = 0;
  private inCols = 0;

  constructor(id: string, poolKind: PoolKind2D = "max", poolSize = 2) {
    super(id, "pool2d", "2d");
    this.poolKind = poolKind;
    this.poolSize = poolSize;
  }

  label(): string {
    return `${this.poolKind === "max" ? "Max" : "Avg"} Pool ${this.poolSize}×${this.poolSize}`;
  }

  outputShape(input: LayerShape): LayerShape {
    if (input.kind !== "2d") throw new Error("Pool2D expects a 2-D input");
    const rows = Math.floor(input.rows / this.poolSize);
    const cols = Math.floor(input.cols / this.poolSize);
    return { kind: "2d", channels: input.channels, rows, cols };
  }

  forward(input: Volume): Volume {
    const channels = input.length;
    this.inRows = input[0].length;
    this.inCols = input[0][0].length;
    const outRows = Math.floor(this.inRows / this.poolSize);
    const outCols = Math.floor(this.inCols / this.poolSize);
    const out = zerosVolume(channels, outRows, outCols);
    this.argmax = new Array(channels);

    for (let c = 0; c < channels; c++) {
      this.argmax[c] = new Array(outRows);
      for (let or = 0; or < outRows; or++) {
        this.argmax[c][or] = new Array(outCols);
        for (let oc = 0; oc < outCols; oc++) {
          const baseR = or * this.poolSize;
          const baseC = oc * this.poolSize;
          if (this.poolKind === "max") {
            let best = -Infinity;
            let bestR = baseR;
            let bestC = baseC;
            for (let kr = 0; kr < this.poolSize; kr++) {
              for (let kc = 0; kc < this.poolSize; kc++) {
                const v = input[c][baseR + kr][baseC + kc];
                if (v > best) {
                  best = v;
                  bestR = baseR + kr;
                  bestC = baseC + kc;
                }
              }
            }
            out[c][or][oc] = best;
            this.argmax[c][or][oc] = { r: bestR, c: bestC };
          } else {
            let sum = 0;
            for (let kr = 0; kr < this.poolSize; kr++) {
              for (let kc = 0; kc < this.poolSize; kc++) {
                sum += input[c][baseR + kr][baseC + kc];
              }
            }
            out[c][or][oc] = sum / (this.poolSize * this.poolSize);
            this.argmax[c][or][oc] = { r: baseR, c: baseC };
          }
        }
      }
    }
    this.output = out;
    return out;
  }

  backward(gradOut: Volume): Volume {
    const channels = gradOut.length;
    const gradIn = zerosVolume(channels, this.inRows, this.inCols);
    const outRows = gradOut[0].length;
    const outCols = gradOut[0][0].length;
    const denom = this.poolSize * this.poolSize;

    for (let c = 0; c < channels; c++) {
      for (let or = 0; or < outRows; or++) {
        for (let oc = 0; oc < outCols; oc++) {
          const g = gradOut[c][or][oc];
          if (this.poolKind === "max") {
            const pos = this.argmax[c][or][oc];
            gradIn[c][pos.r][pos.c] += g;
          } else {
            const each = g / denom;
            const baseR = or * this.poolSize;
            const baseC = oc * this.poolSize;
            for (let kr = 0; kr < this.poolSize; kr++) {
              for (let kc = 0; kc < this.poolSize; kc++) {
                gradIn[c][baseR + kr][baseC + kc] += each;
              }
            }
          }
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

  /** Expose the current activation as a defensive copy (for visualization). */
  snapshotOutput(): Volume {
    return cloneVolume(this.output as Volume);
  }
}
