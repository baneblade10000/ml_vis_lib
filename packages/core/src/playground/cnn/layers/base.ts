import type { Signal, Volume } from "../tensor";
import type { CnnRegularizationId } from "../regularization";
import type { PlaygroundOptimizerId } from "../../optimizers";

/**
 * Discriminated kind used by the UI adapter to map a layer to a React Flow node
 * type and by the engine to group/filter layers.
 */
export type LayerKind =
  | "input"
  | "conv2d"
  | "pool2d"
  | "conv1d"
  | "pool1d"
  | "flatten"
  | "dense"
  | "output";

/** Whether a layer operates on a 2-D {@link Volume} or a 1-D {@link Signal}. */
export type DataSpace = "2d" | "1d";

/**
 * Abstract base for every layer in the convolutional network.
 *
 * Forward pass produces `output`; backward consumes the gradient w.r.t. the
 * output and produces the gradient w.r.t. the input, accumulating parameter
 * gradients internally; {@link updateParams} applies them via plain SGD.
 *
 * A layer is either 2-D (input/output are {@link Volume}) or 1-D
 * (input/output are {@link Signal}). The engine guarantees that adjacent layers
 * agree on {@link dataSpace}; only {@link FlattenLayer} bridges 2-D → 1-D.
 */
export abstract class Layer {
  /** Stable id within the engine (never reused after removal). */
  readonly id: string;
  readonly kind: LayerKind;
  /** Whether forward/backward carry a Volume ("2d") or a Signal ("1d"). */
  readonly dataSpace: DataSpace;

  /** Most recent forward output (read by the UI to render feature maps). */
  output: Volume | Signal = [];
  /** Gradient w.r.t. this layer's input, set by {@link backward}. */
  inputGrad: Volume | Signal = [];

  constructor(id: string, kind: LayerKind, dataSpace: DataSpace) {
    this.id = id;
    this.kind = kind;
    this.dataSpace = dataSpace;
  }

  /** Human-readable label shown on the node (overridable per layer). */
  abstract label(): string;

  /** Compute the output spatial/channel shape from the configured params and the given input shape. */
  abstract outputShape(input: LayerShape): LayerShape;

  /** Forward pass: `input` → `this.output`. Stores reference for visualization. */
  abstract forward(input: Volume | Signal): Volume | Signal;

  /**
   * Backward pass: given `gradOut` (dLoss/dOutput), accumulate parameter grads
   * and set `this.inputGrad` (dLoss/dInput). Returns `inputGrad`.
   */
  abstract backward(gradOut: Volume | Signal): Volume | Signal;

  /**
   * Apply accumulated parameter gradients (SGD / RMSProp / Adam).
   * Weights may take an L1/L2 penalty; biases are never regularized.
   */
  abstract updateParams(
    learningRate: number,
    regularization?: CnnRegularizationId,
    regularizationRate?: number,
    optimizer?: PlaygroundOptimizerId,
    optStep?: number,
  ): void;

  /**
   * Zero the accumulated parameter gradients at the start of a mini-batch. No-op
   * for parameter-less layers. {@link backward} accumulates into these grads.
   */
  abstract zeroGrads(): void;

  /** Total number of trainable parameters (weights + biases). */
  abstract paramCount(): number;

  /** Reset weights/biases to fresh random values, keeping topology. */
  abstract reinitialize(rng: () => number): void;

  /** Aggregated weight magnitude (tanh) for edge colour/width, or null if none. */
  abstract weightMagnitude(): number | null;

  /** Zero Adam/RMSProp moment buffers without touching weights. */
  clearOptimizerState(): void {}

  /**
   * Pack trainable params into `dst` starting at `offset`.
   * Returns the next offset. Default: no params.
   */
  writeParams(_dst: Float64Array, offset: number): number {
    return offset;
  }

  /** Unpack trainable params from `src` starting at `offset`. Returns next offset. */
  readParams(_src: Float64Array, offset: number): number {
    return offset;
  }

  /** Pack accumulated parameter gradients (same layout as params). */
  writeGrads(_dst: Float64Array, offset: number): number {
    return offset;
  }

  /** Replace accumulated grads from a packed buffer (same layout as params). */
  readGrads(_src: Float64Array, offset: number): number {
    return offset;
  }
}

/** Unified shape descriptor for {@link Layer.outputShape}. */
export type LayerShape =
  | { kind: "2d"; channels: number; rows: number; cols: number }
  | { kind: "1d"; channels: number; length: number };

export function shapeSpace(shape: LayerShape): DataSpace {
  return shape.kind;
}

/** Flatten a 2-D volume into a 1-D "single-channel" signal of length channels*rows*cols. */
export function flattenVolume(volume: Volume): Signal {
  const out: number[] = [];
  for (const ch of volume) for (const row of ch) for (const v of row) out.push(v);
  return [out];
}

/** Reshape a flattened single-channel signal back to a volume of the given shape. */
export function unflattenVolume(signal: Signal, channels: number, rows: number, cols: number): Volume {
  const out = new Array(channels);
  let idx = 0;
  for (let c = 0; c < channels; c++) {
    out[c] = new Array(rows);
    for (let r = 0; r < rows; r++) {
      out[c][r] = signal[0].slice(idx, idx + cols);
      idx += cols;
    }
  }
  return out;
}
