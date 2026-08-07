/**
 * Shared rAF-style play loop for train workers (uses worker `self`).
 * Accumulates epoch budget from epochsPerSec and calls `trainOneEpoch`.
 */

export interface PlayLoopHandle {
  play(epochsPerSec: number): void;
  pause(): void;
  /** Run immediately outside the play scheduler. */
  stop(): void;
}

export function createPlayLoop(options: {
  /** Train one epoch; return false to auto-pause. May be async (data-parallel). */
  trainOneEpoch: () => boolean | void | Promise<boolean | void>;
  /** Called after one or more epochs when it's time to publish a viz tick. */
  onPaint: () => void;
  /** Max epochs processed per animation frame. */
  maxEpochsPerFrame?: number;
  paintHz?: number;
}): PlayLoopHandle {
  const maxEpochsPerFrame = options.maxEpochsPerFrame ?? 2;
  const paintIntervalMs = 1000 / (options.paintHz ?? 20);
  let playing = false;
  let epochsPerSec = 12;
  let raf = 0;
  let lastTime = 0;
  let lastPaint = 0;
  let epochBank = 0;
  let busy = false;

  const frame = (now: number) => {
    if (!playing) return;
    if (busy) {
      raf = self.requestAnimationFrame(frame);
      return;
    }
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;
    epochBank = Math.min(epochBank + dt * epochsPerSec, maxEpochsPerFrame);
    const steps = Math.floor(epochBank);
    if (steps <= 0) {
      raf = self.requestAnimationFrame(frame);
      return;
    }
    epochBank -= steps;
    busy = true;
    void (async () => {
      let ran = 0;
      try {
        for (let i = 0; i < steps; i++) {
          const cont = await options.trainOneEpoch();
          ran++;
          if (cont === false) {
            playing = false;
            break;
          }
        }
        if (ran > 0 && performance.now() - lastPaint >= paintIntervalMs) {
          options.onPaint();
          lastPaint = performance.now();
        }
      } finally {
        busy = false;
        if (playing) {
          raf = self.requestAnimationFrame(frame);
        }
      }
    })();
  };

  return {
    play(rate: number) {
      epochsPerSec = rate;
      if (playing) return;
      playing = true;
      lastTime = performance.now();
      lastPaint = 0;
      epochBank = 0;
      raf = self.requestAnimationFrame(frame);
    },
    pause() {
      playing = false;
      if (raf) {
        self.cancelAnimationFrame(raf);
        raf = 0;
      }
    },
    stop() {
      this.pause();
    },
  };
}
