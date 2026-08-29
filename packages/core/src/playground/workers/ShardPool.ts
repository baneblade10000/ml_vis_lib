import { defaultShardCount, sumGradBuffers } from "./shardCount";
import type { FromGradShard, GradShardKind, ToGradShard } from "./shardProtocol";

export interface ShardPoolOptions {
  /** Factory for one grad-shard worker (same URL for all shards). */
  createShardWorker: () => Worker;
  kind: GradShardKind;
  shardCount?: number;
}

type Pending = {
  reject: (err: Error) => void;
  cleanup: () => void;
};

/**
 * Pool of grad-shard workers used by a train coordinator.
 * Coordinator owns Adam; shards only compute gradient sums.
 */
export class ShardPool {
  private workers: Worker[] = [];
  private readonly kind: GradShardKind;
  private readonly createShardWorker: () => Worker;
  readonly shardCount: number;
  private readonly pending = new Set<Pending>();
  /**
   * Latest requested init config — drained by the spawn loop. Rapid
   * re-inits (topology edit bursts) coalesce: only the newest config is
   * spawned, and every caller awaits the same final promise. This also
   * fixes the old hang where a new init's terminateAll() silently orphaned
   * the previous init's per-worker promises (they never settled, so the
   * coordinator's message chain froze forever).
   */
  private latestConfig: unknown = null;
  private spawning = false;
  private initTail: Promise<void> = Promise.resolve();
  private disposed = false;

  constructor(options: ShardPoolOptions) {
    this.kind = options.kind;
    this.createShardWorker = options.createShardWorker;
    this.shardCount = defaultShardCount(options.shardCount);
  }

  /** Spawn shards and init each with the same engine config. */
  init(config: unknown): Promise<void> {
    if (this.disposed) return Promise.resolve();
    this.latestConfig = config;
    if (!this.spawning) {
      this.spawning = true;
      this.initTail = (async () => {
        try {
          while (!this.disposed && this.latestConfig !== null) {
            const cfg = this.latestConfig;
            this.latestConfig = null;
            await this.spawnAll(cfg);
          }
        } finally {
          this.spawning = false;
        }
      })();
    }
    return this.initTail;
  }

  private async spawnAll(config: unknown): Promise<void> {
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
      return this.requestGrads(w, wCopy, shardIdx);
    });
    const results = await Promise.all(tasks);
    const count = results.reduce((s, r) => s + r.count, 0);
    const grads = sumGradBuffers(results.map((r) => r.grads));
    return { grads, count };
  }

  dispose(): void {
    this.disposed = true;
    this.latestConfig = null;
    this.rejectPending("shard pool disposed");
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
    // Critical: in-flight computeGrads must reject, otherwise the play loop's
    // `busy` flag stays true forever after topology rebuild mid-epoch.
    this.rejectPending("shard pool terminated");
    for (const w of this.workers) w.terminate();
    this.workers = [];
  }

  private rejectPending(message: string): void {
    const err = new Error(message);
    for (const p of this.pending) {
      p.cleanup();
      p.reject(err);
    }
    this.pending.clear();
  }

  private requestGrads(
    w: Worker,
    weights: Float64Array,
    indices: number[],
  ): Promise<{ grads: Float64Array; count: number }> {
    return new Promise((resolve, reject) => {
      const onMsg = (ev: MessageEvent<FromGradShard>) => {
        const msg = ev.data;
        if (msg.type === "grads") {
          cleanup();
          this.pending.delete(entry);
          resolve({ grads: msg.grads, count: msg.count });
        } else if (msg.type === "error") {
          cleanup();
          this.pending.delete(entry);
          reject(new Error(msg.message));
        }
      };
      const cleanup = () => w.removeEventListener("message", onMsg);
      const entry: Pending = { reject, cleanup };
      this.pending.add(entry);
      w.addEventListener("message", onMsg);
      try {
        w.postMessage({ type: "compute", weights, indices } satisfies ToGradShard, [weights.buffer]);
      } catch (err) {
        cleanup();
        this.pending.delete(entry);
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }
}

function post(w: Worker, msg: ToGradShard): void {
  w.postMessage(msg);
}
