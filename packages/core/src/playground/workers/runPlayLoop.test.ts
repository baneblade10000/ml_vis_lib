import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPlayLoop } from "./runPlayLoop";

/**
 * `createPlayLoop` is rAF-driven and reads bareword `self.requestAnimationFrame`
 * / `self.cancelAnimationFrame` / `performance.now()`. Node has none of those by
 * default, so we stub them and drive the captured frame callback by hand for
 * full determinism over the epoch budget and paint throttle.
 */

type FrameCb = (now: number) => void;

let perfNow = 0;
let frameClosure: FrameCb | null = null;
let scheduledCount = 0;
let rafId = 0;
const cancels: number[] = [];
let prevSelf: any;

beforeEach(() => {
  perfNow = 1000;
  frameClosure = null;
  scheduledCount = 0;
  rafId = 0;
  cancels.length = 0;

  prevSelf = (globalThis as { self?: unknown }).self;
  // runPlayLoop reads `self.requestAnimationFrame`; alias to globalThis so the
  // stubs below are visible through `self.*`.
  (globalThis as { self: typeof globalThis }).self = globalThis;

  vi.stubGlobal("performance", { now: () => perfNow });
  vi.stubGlobal("requestAnimationFrame", (cb: FrameCb) => {
    frameClosure = cb;
    scheduledCount++;
    return ++rafId;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => {
    cancels.push(id);
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  (globalThis as { self?: unknown }).self = prevSelf;
});

/** Set the clock, then fire the most recently scheduled frame at `now` ms. */
function invokeFrame(now: number): void {
  if (!frameClosure) throw new Error("no frame scheduled");
  perfNow = now;
  frameClosure(now);
}

/** Flush microtasks so the loop's async IIFE settles for sync trainOneEpoch fns. */
async function flush(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

describe("createPlayLoop", () => {
  it("schedules exactly one frame on play(); a second play() only updates rate", () => {
    const handle = createPlayLoop({ trainOneEpoch: () => true, onPaint: () => {} });
    handle.play(10);
    expect(scheduledCount).toBe(1);
    const first = frameClosure;
    expect(first).not.toBeNull();

    handle.play(20); // already playing → early return, no new schedule
    expect(scheduledCount).toBe(1);
    expect(frameClosure).toBe(first);
  });

  it("runs floor(dt*rate) epochs per frame, capped by maxEpochsPerFrame and dt clamp", async () => {
    const train = vi.fn(() => true);
    const handle = createPlayLoop({
      trainOneEpoch: train,
      onPaint: () => {},
      maxEpochsPerFrame: 2,
    });

    handle.play(100); // lastTime = 1000
    invokeFrame(1100); // dt = 0.1s → 0.1*100 = 10, capped to 2
    await flush();
    expect(train).toHaveBeenCalledTimes(2);

    // dt is clamped to 0.1s even for huge time jumps.
    train.mockClear();
    handle.play(1000);
    invokeFrame(1_000_000); // dt clamped to 0.1 → 0.1*1000 = 100, capped to 2
    await flush();
    expect(train).toHaveBeenCalledTimes(2);
  });

  it("accumulates a sub-step epoch budget across frames", async () => {
    const train = vi.fn(() => true);
    const handle = createPlayLoop({
      trainOneEpoch: train,
      onPaint: () => {},
      maxEpochsPerFrame: 2,
    });

    handle.play(10); // lastTime = 1000
    invokeFrame(1050); // dt = 0.05 → budget 0.5 → 0 steps
    await flush();
    expect(train).not.toHaveBeenCalled();

    invokeFrame(1100); // dt = 0.05 → budget 0.5 + 0.5 = 1.0 → 1 step
    await flush();
    expect(train).toHaveBeenCalledTimes(1);
  });

  it("throttles onPaint to paintHz and fires when the interval elapses", async () => {
    const onPaint = vi.fn();
    const train = vi.fn(() => true);
    const handle = createPlayLoop({
      trainOneEpoch: train,
      onPaint,
      maxEpochsPerFrame: 5,
      paintHz: 20, // paintIntervalMs = 50
    });

    handle.play(100); // lastPaint = 0
    invokeFrame(1100); // 1100 - 0 >= 50 → paint
    await flush();
    expect(onPaint).toHaveBeenCalledTimes(1);

    invokeFrame(1130); // 1130 - 1100 = 30 < 50 → throttled
    await flush();
    expect(onPaint).toHaveBeenCalledTimes(1);

    invokeFrame(1160); // 1160 - 1100 = 60 >= 50 → paint
    await flush();
    expect(onPaint).toHaveBeenCalledTimes(2);
  });

  it("auto-pauses when trainOneEpoch returns false and does not reschedule", async () => {
    const train = vi.fn((): boolean => false);
    const onPaint = vi.fn();
    const handle = createPlayLoop({
      trainOneEpoch: train,
      onPaint,
      maxEpochsPerFrame: 5,
    });

    handle.play(100);
    const before = scheduledCount;
    invokeFrame(1100);
    await flush();
    expect(train).toHaveBeenCalledTimes(1);
    expect(scheduledCount).toBe(before); // no next frame scheduled
    expect(cancels).toHaveLength(0);
  });

  it("keeps playing after a throwing epoch (error breaks only that frame)", async () => {
    const train = vi.fn(() => {
      throw new Error("boom");
    });
    const onPaint = vi.fn();
    const handle = createPlayLoop({
      trainOneEpoch: train,
      onPaint,
      maxEpochsPerFrame: 5,
    });

    handle.play(100);
    const before = scheduledCount;
    invokeFrame(1100);
    await flush();
    expect(train).toHaveBeenCalledTimes(1);
    expect(onPaint).not.toHaveBeenCalled(); // ran === 0 → no paint
    expect(scheduledCount).toBe(before + 1); // loop reschedules and continues
  });

  it("does not re-enter trainOneEpoch while an async epoch is in flight (busy guard)", async () => {
    let resolveFirst!: (v: boolean) => void;
    const firstEpoch = new Promise<boolean>((r) => {
      resolveFirst = r;
    });
    let calls = 0;
    const train = vi.fn(() => {
      calls++;
      return calls === 1 ? firstEpoch : Promise.resolve(true);
    });
    const handle = createPlayLoop({
      trainOneEpoch: train,
      onPaint: () => {},
      maxEpochsPerFrame: 2,
    });

    handle.play(100);
    invokeFrame(1100); // starts epoch 0 (deferred) → busy = true
    expect(train).toHaveBeenCalledTimes(1);

    // Spurious frame while busy: must bail and re-request, not start a 2nd epoch.
    const before = scheduledCount;
    invokeFrame(1100);
    expect(train).toHaveBeenCalledTimes(1); // still the one in-flight epoch
    expect(scheduledCount).toBe(before + 1); // re-requested and returned

    resolveFirst(true);
    await flush();
  });

  it("play() resets a stuck busy flag after pause() left an epoch in flight", async () => {
    // First epoch never resolves on its own → busy stays true until reset.
    const train = vi.fn(() => new Promise<boolean>(() => {}));
    const handle = createPlayLoop({
      trainOneEpoch: train,
      onPaint: () => {},
      maxEpochsPerFrame: 2,
    });

    handle.play(100);
    invokeFrame(1100); // busy = true, epoch suspended forever
    await flush();
    expect(train).toHaveBeenCalledTimes(1);

    handle.pause(); // playing = false; busy still true
    expect(cancels.length).toBeGreaterThan(0);

    handle.play(100); // playing was false → proceeds, resets busy, schedules
    const trainBefore = train.mock.calls.length;
    invokeFrame(1200); // busy was reset → frame actually trains (does not bail)
    await flush();
    // The new epoch never resolves, so this frame does not reschedule — but the
    // fact that train was called again proves busy was reset (else it would bail).
    expect(train.mock.calls.length).toBeGreaterThan(trainBefore);
  });

  it("pause() and stop() cancel the scheduled frame and stop scheduling", () => {
    const handle = createPlayLoop({ trainOneEpoch: () => true, onPaint: () => {} });
    handle.play(10);
    expect(scheduledCount).toBe(1);
    const scheduledId = rafId;

    handle.pause();
    expect(cancels).toContain(scheduledId);

    handle.play(10);
    const nextId = rafId;
    handle.stop();
    expect(cancels).toContain(nextId);
  });
});
