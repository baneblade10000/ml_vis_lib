import { cloneSignal, type Signal, type Volume } from "../tensor";
import { Layer, type LayerShape, flattenVolume, unflattenVolume } from "./base";

/**
 * Reshape an activation into a single-channel {@link Signal} (a 1-D vector), so
 * it can feed a {@link DenseLayer}. Bridges two cases:
 *
 *  - **2-D → 1-D:** a {@link Volume} (channels × rows × cols) is laid out
 *    channel-major, then row, then column into one vector of length
 *    `channels · rows · cols`. The backward pass reshapes the 1-D gradient back
 *    into the original volume.
 *  - **1-D → 1-D:** a multi-channel {@link Signal} (channels × length) is
 *    concatenated channel-major into a single vector of length
 *    `channels · length`. The backward pass reshapes it back into channels.
 */
export class FlattenLayer extends Layer {
  /** 2-D source shape (set on forward when the input is a Volume). */
  private srcChannels = 0;
  private srcRows = 0;
  private srcCols = 0;
  /** 1-D source shape (set on forward when the input is a Signal). */
  private srcSigChannels = 0;
  private srcLen = 0;
  private wasVolume = false;

  constructor(id: string) {
    // Output is always a 1-D single-channel signal.
    super(id, "flatten", "1d");
  }

  label(): string {
    return "Flatten";
  }

  outputShape(input: LayerShape): LayerShape {
    if (input.kind === "2d") {
      return { kind: "1d", channels: 1, length: input.channels * input.rows * input.cols };
    }
    return { kind: "1d", channels: 1, length: input.channels * input.length };
  }

  forward(input: Volume | Signal): Signal {
    const sig = input as Signal;
    // A Volume is `number[][][]` whose innermost element is a number; a
    // multi-channel Signal is `number[][]` whose element is a number. Distinguish
    // by checking whether the first leaf is an array (Volume) or a number (Signal).
    this.wasVolume = input.length > 0 && Array.isArray((sig[0] as number[])[0]);
    if (this.wasVolume) {
      const vol = input as Volume;
      this.srcChannels = vol.length;
      this.srcRows = vol[0].length;
      this.srcCols = vol[0][0].length;
      this.output = flattenVolume(vol);
    } else {
      const multi = input as Signal;
      this.srcSigChannels = multi.length;
      this.srcLen = multi[0].length;
      const out: number[] = [];
      for (let c = 0; c < multi.length; c++) for (let p = 0; p < multi[c].length; p++) out.push(multi[c][p]);
      this.output = [out];
    }
    return this.output as Signal;
  }

  backward(gradOut: Volume | Signal): Volume | Signal {
    const grad = gradOut as Signal;
    if (this.wasVolume) {
      const gradVolume = unflattenVolume(grad, this.srcChannels, this.srcRows, this.srcCols);
      this.inputGrad = gradVolume;
      return gradVolume;
    }
    // 1-D → 1-D: split the single vector back into channels.
    const out = new Array(this.srcSigChannels);
    const flat = grad[0];
    let idx = 0;
    for (let c = 0; c < this.srcSigChannels; c++) {
      out[c] = flat.slice(idx, idx + this.srcLen);
      idx += this.srcLen;
    }
    this.inputGrad = out;
    return out;
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
