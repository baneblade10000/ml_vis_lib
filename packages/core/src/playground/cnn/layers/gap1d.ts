import { cloneSignal, zerosSignal, type Signal } from "../tensor";
import { Layer, type LayerShape } from "./base";

/**
 * Global average pooling over the temporal axis: each channel → one scalar mean.
 * 1-D counterpart of {@link GlobalAvgPool2DLayer}.
 */
export class GlobalAvgPool1DLayer extends Layer {
  private inChannels = 0;
  private inLen = 0;

  constructor(id: string) {
    super(id, "gap1d", "1d");
  }

  label(): string {
    return "Global Avg Pool";
  }

  outputShape(input: LayerShape): LayerShape {
    if (input.kind !== "1d") throw new Error("GlobalAvgPool1D expects a 1-D input");
    return { kind: "1d", channels: 1, length: input.channels };
  }

  forward(input: Signal): Signal {
    this.inChannels = input.length;
    this.inLen = input[0]?.length ?? 0;
    const denom = Math.max(1, this.inLen);
    const out = new Array<number>(this.inChannels);
    for (let c = 0; c < this.inChannels; c++) {
      let sum = 0;
      const row = input[c]!;
      for (let p = 0; p < this.inLen; p++) sum += row[p]!;
      out[c] = sum / denom;
    }
    this.output = [out];
    return this.output as Signal;
  }

  backward(gradOut: Signal): Signal {
    const g = gradOut[0] ?? [];
    const denom = Math.max(1, this.inLen);
    const gradIn = zerosSignal(this.inChannels, this.inLen);
    for (let c = 0; c < this.inChannels; c++) {
      const each = (g[c] ?? 0) / denom;
      for (let p = 0; p < this.inLen; p++) gradIn[c]![p]! += each;
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
