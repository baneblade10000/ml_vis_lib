import { describe, expect, it } from "vitest";
import { ShardPool } from "./ShardPool";
import type { FromGradShard, GradShardKind, ToGradShard } from "./shardProtocol";
import { FakeWorker } from "./test-helpers/FakeWorker";

const KIND: GradShardKind = "network";

/** The pool expects a real `Worker`; FakeWorker is a structural stand-in. */
function asWorker(w: FakeWorker): Worker {
  return w as unknown as Worker;
}

/** A FakeWorker that acknowledges init and returns deterministic grads per shard. */
function makeResponsiveWorker(): FakeWorker {
  const w = new FakeWorker();
  w.onPost = (msg) => {
    const m = msg as ToGradShard;
    if (m.type === "init") {
      w.emit({ type: "ready" } satisfies FromGradShard);
    } else if (m.type === "compute") {
      const count = m.indices.length;
      // All weight slots get this shard's count → summed buffer ≡ total count.
      const grads = new Float64Array(m.weights.length).fill(count);
      w.emit({ type: "grads", grads, count } satisfies FromGradShard);
    }
  };
  return w;
}

describe("ShardPool", () => {
  it("init() spawns no workers when shardCount <= 1", async () => {
    let spawned = 0;
    const pool = new ShardPool({
      createShardWorker: () => {
        spawned++;
        return asWorker(new FakeWorker());
      },
      kind: KIND,
      shardCount: 1,
    });
    await pool.init({ any: "config" });
    expect(spawned).toBe(0);
    expect(pool.shardCount).toBe(1);
  });

  it("init() spawns N workers and waits for ready from each", async () => {
    const workers: FakeWorker[] = [];
    const pool = new ShardPool({
      createShardWorker: () => {
        const w = makeResponsiveWorker();
        workers.push(w);
        return asWorker(w);
      },
      kind: KIND,
      shardCount: 3,
    });
    await pool.init({ any: "config" });
    expect(workers).toHaveLength(3);
    for (const w of workers) {
      const init = w.messages[0] as ToGradShard;
      expect(init.type).toBe("init");
    }
  });

  it("init() rejects when a shard reports error instead of ready", async () => {
    const pool = new ShardPool({
      createShardWorker: () => {
        const w = new FakeWorker();
        w.onPost = (msg) => {
          if ((msg as ToGradShard).type === "init") {
            w.emit({ type: "error", message: "nope" } satisfies FromGradShard);
          }
        };
        return asWorker(w);
      },
      kind: KIND,
      shardCount: 2,
    });
    await expect(pool.init({})).rejects.toThrow("nope");
  });

  it("computeGrads() returns null before init or with empty indices", async () => {
    const pool = new ShardPool({
      createShardWorker: () => asWorker(makeResponsiveWorker()),
      kind: KIND,
      shardCount: 2,
    });
    // No init yet → no workers.
    expect(await pool.computeGrads(new Float64Array(4), [0, 1])).toBeNull();
    await pool.init({});
    expect(await pool.computeGrads(new Float64Array(4), [])).toBeNull();
  });

  it("computeGrads() partitions indices round-robin, sums grads and counts", async () => {
    const seen: number[][] = [];
    const pool = new ShardPool({
      createShardWorker: () => {
        const w = new FakeWorker();
        w.onPost = (msg) => {
          const m = msg as ToGradShard;
          if (m.type === "init") {
            w.emit({ type: "ready" } satisfies FromGradShard);
          } else if (m.type === "compute") {
            seen.push([...m.indices]);
            const grads = new Float64Array(m.weights.length).fill(m.indices.length);
            w.emit({
              type: "grads",
              grads,
              count: m.indices.length,
            } satisfies FromGradShard);
          }
        };
        return asWorker(w);
      },
      kind: KIND,
      shardCount: 2,
    });
    await pool.init({});

    const weights = new Float64Array(5);
    const result = await pool.computeGrads(weights, [0, 1, 2, 3, 4, 5]);
    expect(result).not.toBeNull();
    // Round-robin: worker0 → [0,2,4], worker1 → [1,3,5] (order may vary by resolve).
    const flat = seen.flat().sort((a, b) => a - b);
    expect(flat).toEqual([0, 1, 2, 3, 4, 5]);
    expect(seen).toHaveLength(2);
    expect(result!.count).toBe(6);
    expect(Array.from(result!.grads)).toEqual([6, 6, 6, 6, 6]);
  });

  it("computeGrads() rejects when a shard returns error", async () => {
    const pool = new ShardPool({
      createShardWorker: () => {
        const w = new FakeWorker();
        w.onPost = (msg) => {
          const m = msg as ToGradShard;
          if (m.type === "init") {
            w.emit({ type: "ready" } satisfies FromGradShard);
          } else if (m.type === "compute") {
            w.emit({ type: "error", message: "shard failed" } satisfies FromGradShard);
          }
        };
        return asWorker(w);
      },
      kind: KIND,
      shardCount: 2,
    });
    await pool.init({});
    await expect(pool.computeGrads(new Float64Array(3), [0, 1, 2])).rejects.toThrow(
      "shard failed",
    );
  });

  it("dispose() posts dispose to every worker, terminates them, and clears the pool", async () => {
    const workers: FakeWorker[] = [];
    const pool = new ShardPool({
      createShardWorker: () => {
        const w = makeResponsiveWorker();
        workers.push(w);
        return asWorker(w);
      },
      kind: KIND,
      shardCount: 3,
    });
    await pool.init({});
    pool.dispose();
    for (const w of workers) {
      const last = w.messages.at(-1) as ToGradShard;
      expect(last.type).toBe("dispose");
      expect(w.terminatedCount).toBe(1);
    }
    // Pool is now empty.
    expect(await pool.computeGrads(new Float64Array(2), [0])).toBeNull();
    // Second dispose is a no-op (does not throw).
    expect(() => pool.dispose()).not.toThrow();
  });

  it("re-init rejects in-flight computeGrads (the busy-flag unblock path)", async () => {
    const pool = new ShardPool({
      createShardWorker: () => {
        const w = new FakeWorker();
        w.onPost = (msg) => {
          if ((msg as ToGradShard).type === "init") {
            w.emit({ type: "ready" } satisfies FromGradShard);
          }
          // "compute" intentionally unanswered.
        };
        return asWorker(w);
      },
      kind: KIND,
      shardCount: 2,
    });
    await pool.init({});

    const pending = pool.computeGrads(new Float64Array(3), [0, 1, 2]);
    // Re-init tears down the pool mid-flight → in-flight grads must reject.
    // (Reject happens in the synchronous terminateAll() at the start of init().)
    const reInit = pool.init({});
    await expect(pending).rejects.toThrow();
    await reInit; // safe to re-init afterwards
  });

  it("computeGrads() rejects cleanly when postMessage throws", async () => {
    const pool = new ShardPool({
      createShardWorker: () => {
        const w = new FakeWorker();
        w.onPost = (msg) => {
          if ((msg as ToGradShard).type === "init") {
            w.emit({ type: "ready" } satisfies FromGradShard);
          }
        };
        // Force the compute post to throw (e.g. detached buffer / worker gone).
        const orig = w.postMessage.bind(w);
        w.postMessage = ((msg: unknown) => {
          if ((msg as ToGradShard).type === "compute") {
            throw new Error("postMessage boom");
          }
          orig(msg);
        }) as typeof w.postMessage;
        return asWorker(w);
      },
      kind: KIND,
      shardCount: 2,
    });
    await pool.init({});
    await expect(pool.computeGrads(new Float64Array(3), [0, 1, 2])).rejects.toThrow(
      "postMessage boom",
    );
    pool.dispose();
  });
});
