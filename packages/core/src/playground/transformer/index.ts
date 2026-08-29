/**
 * Transformer playground — shared DTO types.
 *
 * All compute (forward/backward/Adam/greedy decode) lives in the Rust crate
 * `crates/transformer` compiled to WASM; this module only mirrors the snapshot
 * JSON the worker posts to the UI (`crates/transformer/src/snapshot.rs`).
 */

export type TransformerTaskId = "translate" | "reverse";

export const TRANSFORMER_TASKS: TransformerTaskId[] = ["translate", "reverse"];

export function isTransformerTaskId(id: string): id is TransformerTaskId {
  return (TRANSFORMER_TASKS as string[]).includes(id);
}

export interface TransformerConfigDto {
  alphabetSize: number;
  dModel: number;
  heads: number;
  encLayers: number;
  decLayers: number;
  ffDim: number;
  maxLen: number;
  learningRate: number;
  seed: number;
}

/** Mirrors `TransformerConfig::default()` in Rust. */
export const DEFAULT_TRANSFORMER_CONFIG: TransformerConfigDto = {
  alphabetSize: 20,
  dModel: 32,
  heads: 4,
  encLayers: 2,
  decLayers: 2,
  ffDim: 64,
  maxLen: 6,
  learningRate: 0.003,
  seed: 42,
};

/** Attention probabilities `[head][row][col]` for one layer. */
export type HeadMatrices = number[][][];

export interface AttentionDto {
  encSelf: HeadMatrices[];
  decSelf: HeadMatrices[];
  cross: HeadMatrices[];
}

export interface LossPoint {
  step: number;
  loss: number;
}

export interface TransformerSnapshot {
  step: number;
  /** Exponential moving average of the training loss. */
  loss: number;
  /** Token-level accuracy of the greedy decode on the display sample. */
  accuracy: number;
  task: string;
  inputTokens: number[];
  targetTokens: number[];
  predictedTokens: number[];
  /** Decoder input tokens (`<s>` + target) aligned with decSelf/cross rows. */
  decInTokens: number[];
  /** Encoder input tokens (input + `</s>`) aligned with encSelf/cross columns. */
  encInTokens: number[];
  attention: AttentionDto;
  lossHistory: LossPoint[];
  alphabetSize: number;
  /** Display label per token id (Russian words, English words, `<s>`, `</s>`). */
  labels: string[];
}
