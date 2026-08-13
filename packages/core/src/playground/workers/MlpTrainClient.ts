/**
 * Type-safe wrapper around {@link TrainWorkerClient} for the legacy 2D MLP
 * decision-boundary engine. The MLP worker exposes no `command` channel and only
 * ever rebuilds with reason `"reset"`, so this wrapper is intentionally small.
 */

import type { PlaygroundConfig } from "../types";
import type { MlpTrainSnapshot, TrainSnapshot } from "./protocol";
import { TrainWorkerClient } from "./TrainWorkerClient";

export interface MlpTrainClientOptions {
  createWorker?: () => Worker;
  client?: TrainWorkerClient;
  onTick: (snapshot: MlpTrainSnapshot) => void;
  onError?: (message: string) => void;
}

export class MlpTrainClient {
  private readonly client: TrainWorkerClient;
  private readonly onTick: (snapshot: MlpTrainSnapshot) => void;

  constructor(options: MlpTrainClientOptions) {
    this.onTick = options.onTick;
    const deliver = (s: TrainSnapshot) => {
      if (s.kind === "mlp") this.onTick(s);
    };
    if (options.client) {
      this.client = options.client;
      this.client.setOnTick(deliver);
      if (options.onError) this.client.setOnError(options.onError);
    } else if (options.createWorker) {
      this.client = new TrainWorkerClient({
        createWorker: options.createWorker,
        onTick: deliver,
        onError: options.onError,
      });
    } else {
      throw new Error("MlpTrainClient requires either createWorker or client");
    }
  }

  async init(config: PlaygroundConfig): Promise<MlpTrainSnapshot> {
    const s = await this.client.init(config);
    return assertMlp(s);
  }

  /** The MLP worker only rebuilds on full reset (optionally with a new config). */
  rebuild(reason: "reset", payload?: PlaygroundConfig): Promise<void> {
    return this.client.rebuild(reason, payload);
  }

  play(epochsPerSec: number): void {
    this.client.play(epochsPerSec);
  }

  pause(): void {
    this.client.pause();
  }

  step(): void {
    this.client.step();
  }

  dispose(): void {
    this.client.dispose();
  }

  get snapshot(): MlpTrainSnapshot | null {
    const s = this.client.snapshot;
    return s && s.kind === "mlp" ? s : null;
  }
}

function assertMlp(s: TrainSnapshot): MlpTrainSnapshot {
  if (s.kind !== "mlp") {
    throw new Error(`MlpTrainClient expected an mlp snapshot, got "${s.kind}"`);
  }
  return s;
}
