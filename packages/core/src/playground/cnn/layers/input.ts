import { cloneSignal, cloneVolume, type Signal, type Volume } from "../tensor";
import { Layer, type LayerShape } from "./base";

/**
 * Input layer — a pass-through that just stores the current example and reports
 * its shape. It exists so the network graph has an explicit first node and so
 * the UI can render the raw input as a feature map.
 */
export class InputLayer extends Layer {
  readonly channels: number;
  /** Fixed spatial size; `null` for 1-D input. */
  readonly rows: number | null;
  readonly cols: number | null;
  readonly length: number | null;

  constructor(
    id: string,
    dataSpace: "2d" | "1d",
    dims: { channels: number; rows: number; cols: number } | { channels: number; length: number },
  ) {
    super(id, "input", dataSpace);
    if (dataSpace === "2d") {
      const d = dims as { channels: number; rows: number; cols: number };
      this.channels = d.channels;
      this.rows = d.rows;
      this.cols = d.cols;
      this.length = null;
    } else {
      const d = dims as { channels: number; length: number };
      this.channels = d.channels;
      this.length = d.length;
      this.rows = null;
      this.cols = null;
    }
  }

  label(): string {
    if (this.dataSpace === "2d") return `Input ${this.rows}×${this.cols}`;
    return `Input ${this.length}`;
  }

  outputShape(_input: LayerShape): LayerShape {
    if (this.dataSpace === "2d") {
      return { kind: "2d", channels: this.channels, rows: this.rows!, cols: this.cols! };
    }
    return { kind: "1d", channels: this.channels, length: this.length! };
  }

  forward(input: Volume | Signal): Volume | Signal {
    this.output = this.dataSpace === "2d" ? cloneVolume(input as Volume) : cloneSignal(input as Signal);
    return this.output;
  }

  backward(gradOut: Volume | Signal): Volume | Signal {
    this.inputGrad = this.dataSpace === "2d" ? cloneVolume(gradOut as Volume) : cloneSignal(gradOut as Signal);
    return this.inputGrad;
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
}
