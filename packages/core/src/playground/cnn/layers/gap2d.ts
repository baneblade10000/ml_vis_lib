import { acquireVolume, cloneSignal, type Signal, type Volume } from "../tensor";
import { Layer, type LayerShape } from "./base";

/**
 * Global average pooling over spatial dims: each channel → one scalar mean.
 * Bridges 2-D feature maps to a dense head without a full Flatten.
 */
export class GlobalAvgPool2DLayer extends Layer {
  private inChannels = 0;
  private inRows = 0;
  private inCols = 0;
  private scratchOut: number[] = [];
  private scratchGradIn: Volume = [];

  constructor(id: string) {
    super(id, "gap2d", "1d");
  }

  label(): string {
    return "Global Avg Pool";
  }

  outputShape(input: LayerShape): LayerShape {
    if (input.kind !== "2d") throw new Error("GlobalAvgPool2D expects a 2-D input");
    return { kind: "1d", channels: 1, length: input.channels };
  }

  forward(input: Volume): Signal {
    this.inChannels = input.length;
    this.inRows = input[0]?.length ?? 0;
    this.inCols = input[0]?.[0]?.length ?? 0;
    const denom = Math.max(1, this.inRows * this.inCols);
    if (this.scratchOut.length !== this.inChannels) {
      this.scratchOut = new Array(this.inChannels);
    }
    const out = this.scratchOut;
    for (let c = 0; c < this.inChannels; c++) {
      let sum = 0;
      const ch = input[c]!;
      for (let r = 0; r < this.inRows; r++) {
        const row = ch[r]!;
        for (let col = 0; col < this.inCols; col++) sum += row[col]!;
      }
      out[c] = sum / denom;
    }
    this.output = [out];
    return this.output as Signal;
  }

  backward(gradOut: Signal): Volume {
    const g = gradOut[0] ?? [];
    const denom = Math.max(1, this.inRows * this.inCols);
    const gradIn = acquireVolume(this.scratchGradIn, this.inChannels, this.inRows, this.inCols);
    this.scratchGradIn = gradIn;
    for (let c = 0; c < this.inChannels; c++) {
      const each = (g[c] ?? 0) / denom;
      const ch = gradIn[c]!;
      for (let r = 0; r < this.inRows; r++) {
        const row = ch[r]!;
        for (let col = 0; col < this.inCols; col++) row[col] = each;
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
