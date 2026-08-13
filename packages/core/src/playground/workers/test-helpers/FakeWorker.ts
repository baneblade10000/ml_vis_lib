/**
 * Minimal in-memory Worker stub for testing worker-owning code (ShardPool,
 * TrainWorkerClient, typed train clients) without a real Web Worker.
 *
 * Tests drive it by calling `emit(...)` to deliver messages; `messages` records
 * everything the owning code posted so call sites can assert on the protocol.
 */
export type WorkerEventListener = (ev: { data: unknown }) => void;

export class FakeWorker {
  private listeners = new Map<string, Set<WorkerEventListener>>();
  /** Every message posted to this worker, in order. */
  messages: unknown[] = [];
  /** Number of times `terminate()` was called. */
  terminatedCount = 0;
  /** Assigned by owners (e.g. ShardPool.init) — matches the real Worker surface. */
  onerror: ((ev: { message: string }) => void) | null = null;
  /** `onmessage` handler property (TrainWorkerClient uses this, not addEventListener). */
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  /** Optional hook to react to an incoming post synchronously (auto-respond). */
  onPost: ((msg: unknown, transfer: Transferable[] | undefined) => void) | null = null;

  addEventListener(type: string, cb: WorkerEventListener): void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(cb);
  }

  removeEventListener(type: string, cb: WorkerEventListener): void {
    this.listeners.get(type)?.delete(cb);
  }

  postMessage(msg: unknown, transfer?: Transferable[] | undefined): void {
    this.messages.push(msg);
    this.onPost?.(msg, transfer);
  }

  terminate(): void {
    this.terminatedCount++;
  }

  /** Deliver a message to the onmessage handler and all "message" listeners. */
  emit(data: unknown): void {
    const ev = { data };
    this.onmessage?.(ev);
    const set = this.listeners.get("message");
    if (set) for (const cb of set) cb(ev);
  }
}
