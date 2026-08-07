import { acquireVolume, cloneVolume, type Volume } from "../tensor";
import { Layer, type LayerShape } from "./base";

export type PoolKind2D = "max" | "avg";

/**
 * 2×2 pooling with stride 2. No learnable params — forward remembers the argmax
 * positions (for `max`) so the gradient can be routed to exactly those cells.
 */
export class Pool2DLayer extends Layer {
  readonly poolSize: number;
  readonly poolKind: PoolKind2D;
  /** Flat argmax row/col per output cell: index = ((c*outRows)+or)*outCols+oc. */
  private argmaxR: Int32Array = new Int32Array(0);
  private argmaxC: Int32Array = new Int32Array(0);
  private outRows = 0;
  private outCols = 0;
  private inRows = 0;
  private inCols = 0;
  private scratchGradIn: Volume = [];

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
    this.outRows = outRows;
    this.outCols = outCols;
    const out = acquireVolume((this.output as Volume) ?? [], channels, outRows, outCols);
    const cells = channels * outRows * outCols;
    if (this.argmaxR.length !== cells) {
      this.argmaxR = new Int32Array(cells);
      this.argmaxC = new Int32Array(cells);
    }
    const ps = this.poolSize;
    const isMax = this.poolKind === "max";
    const denom = ps * ps;

    for (let c = 0; c < channels; c++) {
      const inCh = input[c]!;
      const outCh = out[c]!;
      const baseIdx = c * outRows * outCols;
      for (let or = 0; or < outRows; or++) {
        for (let oc = 0; oc < outCols; oc++) {
          const baseR = or * ps;
          const baseC = oc * ps;
          const idx = baseIdx + or * outCols + oc;
          if (isMax) {
            let best = -Infinity;
            let bestR = baseR;
            let bestC = baseC;
            for (let kr = 0; kr < ps; kr++) {
              const row = inCh[baseR + kr]!;
              for (let kc = 0; kc < ps; kc++) {
                const v = row[baseC + kc]!;
                if (v > best) {
                  best = v;
                  bestR = baseR + kr;
                  bestC = baseC + kc;
                }
              }
            }
            outCh[or]![oc] = best;
            this.argmaxR[idx] = bestR;
            this.argmaxC[idx] = bestC;
          } else {
            let sum = 0;
            for (let kr = 0; kr < ps; kr++) {
              const row = inCh[baseR + kr]!;
              for (let kc = 0; kc < ps; kc++) sum += row[baseC + kc]!;
            }
            outCh[or]![oc] = sum / denom;
            this.argmaxR[idx] = baseR;
            this.argmaxC[idx] = baseC;
          }
        }
      }
    }
    this.output = out;
    return out;
  }

  backward(gradOut: Volume): Volume {
    const channels = gradOut.length;
    const gradIn = acquireVolume(this.scratchGradIn, channels, this.inRows, this.inCols);
    this.scratchGradIn = gradIn;
    const outRows = this.outRows;
    const outCols = this.outCols;
    const denom = this.poolSize * this.poolSize;
    const ps = this.poolSize;
    const isMax = this.poolKind === "max";

    for (let c = 0; c < channels; c++) {
      const gCh = gradOut[c]!;
      const inCh = gradIn[c]!;
      const baseIdx = c * outRows * outCols;
      for (let or = 0; or < outRows; or++) {
        for (let oc = 0; oc < outCols; oc++) {
          const g = gCh[or]![oc]!;
          const idx = baseIdx + or * outCols + oc;
          if (isMax) {
            inCh[this.argmaxR[idx]!]![this.argmaxC[idx]!]! += g;
          } else {
            const each = g / denom;
            const baseR = or * ps;
            const baseC = oc * ps;
            for (let kr = 0; kr < ps; kr++) {
              const row = inCh[baseR + kr]!;
              for (let kc = 0; kc < ps; kc++) row[baseC + kc]! += each;
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
