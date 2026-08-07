import type { FromTrainWorker, ToTrainWorker, TrainRebuildReason, TrainSnapshot } from "./protocol";

/** True when the environment can spawn module workers. */
export function canUseTrainWorkers(): boolean {
  return typeof Worker !== "undefined";
}

export type TrainWorkerTickHandler = (snapshot: TrainSnapshot) => void;
export type TrainWorkerErrorHandler = (message: string) => void;

export interface TrainWorkerClientOptions {
  /** Factory that returns a Worker (or a Worker-like shim). */
  createWorker: () => Worker;
  onTick?: TrainWorkerTickHandler;
  onError?: TrainWorkerErrorHandler;
}

/**
 * Main-thread client for a playground train worker.
 * Posts commands; delivers ready/tick snapshots via callbacks.
 */
export class TrainWorkerClient {
  private worker: Worker | null = null;
  private readyPromise: Promise<TrainSnapshot> | null = null;
  private readyResolve: ((s: TrainSnapshot) => void) | null = null;
  private disposed = false;
  private onTick: TrainWorkerTickHandler | undefined;
  private onError: TrainWorkerErrorHandler | undefined;
  /** Latest snapshot (ready or tick). */
  snapshot: TrainSnapshot | null = null;

  constructor(private readonly options: TrainWorkerClientOptions) {
    this.onTick = options.onTick;
    this.onError = options.onError;
  }

  setOnTick(handler: TrainWorkerTickHandler | undefined): void {
    this.onTick = handler;
  }

  setOnError(handler: TrainWorkerErrorHandler | undefined): void {
    this.onError = handler;
  }

  async init(config: unknown): Promise<TrainSnapshot> {
    this.disposeWorkerOnly();
    this.disposed = false;
    this.worker = this.options.createWorker();
    this.worker.onmessage = (ev: MessageEvent<FromTrainWorker>) => this.handleMessage(ev.data);
    this.worker.onerror = (ev) => {
      const message = ev.message || "Train worker error";
      this.onError?.(message);
    };
    this.readyPromise = new Promise<TrainSnapshot>((resolve) => {
      this.readyResolve = resolve;
    });
    this.post({ type: "init", config });
    return this.readyPromise;
  }

  play(epochsPerSec: number): void {
    this.post({ type: "play", epochsPerSec });
  }

  pause(): void {
    this.post({ type: "pause" });
  }

  step(): void {
    this.post({ type: "step" });
  }

  setConfig(patch: Record<string, unknown>): void {
    this.post({ type: "setConfig", patch });
  }

  rebuild(reason: TrainRebuildReason, payload?: unknown): void {
    this.post({ type: "rebuild", reason, payload });
  }

  inspect(exampleIndex?: number): void {
    this.post({ type: "inspect", exampleIndex });
  }

  /** Engine-specific imperative commands (e.g. setFilters). */
  command(name: string, args?: unknown): void {
    this.post({ type: "command", name, args });
  }

  dispose(): void {
    this.disposed = true;
    if (this.worker) {
      this.post({ type: "dispose" });
    }
    this.disposeWorkerOnly();
  }

  private disposeWorkerOnly(): void {
    if (this.worker) {
      this.worker.onmessage = null;
      this.worker.onerror = null;
      this.worker.terminate();
      this.worker = null;
    }
    this.readyPromise = null;
    this.readyResolve = null;
  }

  private post(msg: ToTrainWorker): void {
    if (!this.worker || this.disposed) return;
    try {
      this.worker.postMessage(msg);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.onError?.(message);
    }
  }

  private handleMessage(msg: FromTrainWorker): void {
    if (this.disposed) return;
    if (msg.type === "error") {
      this.onError?.(msg.message);
      return;
    }
    this.snapshot = msg.snapshot;
    if (msg.type === "ready" && this.readyResolve) {
      this.readyResolve(msg.snapshot);
      this.readyResolve = null;
    }
    this.onTick?.(msg.snapshot);
  }
}
