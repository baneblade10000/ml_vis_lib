import { defaultShardCount, sumGradBuffers } from "./shardCount";
import type { FromGradShard, GradShardKind, ToGradShard } from "./shardProtocol";

export interface ShardPoolOptions {
  /** Factory for one grad-shard worker (same URL for all shards). */
  createShardWorker: () => Worker;
  kind: GradShardKind;
  shardCount?: number;
}

/**
 * Pool of grad-shard workers used by a train coordinator.
 * Coordinator owns Adam; shards only compute gradient sums.
 */
export class ShardPool {
  private workers: Worker[] = [];
  private readonly kind: GradShardKind;
  private readonly createShardWorker: () => Worker;
  readonly shardCount: number;

  constructor(options: ShardPoolOptions) {
    this.kind = options.kind;
    this.createShardWorker = options.createShardWorker;
    this.shardCount = defaultShardCount(options.shardCount);
  }

  /** Spawn shards and init each with the same engine config. */
  async init(config: unknown): Promise<void> {
    this.terminateAll();
    if (this.shardCount <= 1) return;
    const ready: Promise<void>[] = [];
    for (let i = 0; i < this.shardCount; i++) {
      const w = this.createShardWorker();
      this.workers.push(w);
      ready.push(
        new Promise<void>((resolve, reject) => {
          const onMsg = (ev: MessageEvent<FromGradShard>) => {
            if (ev.data.type === "ready") {
              w.removeEventListener("message", onMsg);
              resolve();
            } else if (ev.data.type === "error") {
              w.removeEventListener("message", onMsg);
              reject(new Error(ev.data.message));
            }
          };
          w.addEventListener("message", onMsg);
          w.onerror = (ev) => reject(new Error(ev.message || "shard worker error"));
          post(w, { type: "init", kind: this.kind, config });
        }),
      );
    }
    await Promise.all(ready);
  }

  /** Replace each shard's train-data copy (serializable examples / features). */
  setTrainData(data: unknown): void {
    for (const w of this.workers) {
      post(w, { type: "setTrainData", data });
    }
  }

  /**
   * Broadcast weights; partition `indices` across shards; return summed grads.
   * When shardCount≤1 or pool empty, returns null (use local engine path).
   */
  async computeGrads(
    weights: Float64Array,
    indices: number[],
  ): Promise<{ grads: Float64Array; count: number } | null> {
    if (this.workers.length === 0 || indices.length === 0) return null;
    const n = this.workers.length;
    const tasks = this.workers.map((w, i) => {
      const shardIdx: number[] = [];
      for (let k = i; k < indices.length; k += n) shardIdx.push(indices[k]!);
      const wCopy = weights.slice();
      return requestGrads(w, wCopy, shardIdx);
    });
    const results = await Promise.all(tasks);
    const count = results.reduce((s, r) => s + r.count, 0);
    const grads = sumGradBuffers(results.map((r) => r.grads));
    return { grads, count };
  }

  dispose(): void {
    for (const w of this.workers) {
      try {
        post(w, { type: "dispose" });
      } catch {
        /* ignore */
      }
      w.terminate();
    }
    this.workers = [];
  }

  private terminateAll(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
  }
}

function post(w: Worker, msg: ToGradShard): void {
  w.postMessage(msg);
}

function requestGrads(
  w: Worker,
  weights: Float64Array,
  indices: number[],
): Promise<{ grads: Float64Array; count: number }> {
  return new Promise((resolve, reject) => {
    const onMsg = (ev: MessageEvent<FromGradShard>) => {
      const msg = ev.data;
      if (msg.type === "grads") {
        w.removeEventListener("message", onMsg);
        resolve({ grads: msg.grads, count: msg.count });
      } else if (msg.type === "error") {
        w.removeEventListener("message", onMsg);
        reject(new Error(msg.message));
      }
    };
    w.addEventListener("message", onMsg);
    w.postMessage({ type: "compute", weights, indices } satisfies ToGradShard, [weights.buffer]);
  });
}
