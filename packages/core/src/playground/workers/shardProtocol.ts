/**
 * Messages between a train coordinator worker and its grad-shard workers.
 */

export type GradShardKind = "cnn" | "network" | "mlp";

export type ToGradShard =
  | { type: "init"; kind: GradShardKind; config: unknown }
  | {
      type: "compute";
      weights: Float64Array;
      /** Indices into the shard's local train-data copy. */
      indices: number[];
    }
  | { type: "setTrainData"; data: unknown }
  | { type: "dispose" };

export type FromGradShard =
  | { type: "ready" }
  | { type: "grads"; grads: Float64Array; count: number }
  | { type: "error"; message: string };
