/**
 * Type-safe wrapper around {@link TrainWorkerClient} for the CNN (CNN WASM) engine.
 *
 * `TrainWorkerClient` is an intentionally untyped transport (the shared worker
 * protocol is engine-agnostic). This wrapper re-adds the per-engine types that
 * matter on the main thread: a concrete config, reason→payload overloads for
 * `rebuild`, a typed `command` channel, and an `onTick` narrowed to
 * {@link CnnTrainSnapshot} so consumers no longer need their own `kind` guards.
 *
 * The worker side keeps its own `unknown`-casts (crossing into the live engine
 * is a separate boundary); this wrapper only protects the main-thread surface.
 */

import type { CnnActivationId } from "../cnn/activations";
import type { CnnConfig, CnnMode } from "../cnn/engine";
import type { CnnRegularizationId } from "../cnn/regularization";
import type { PlaygroundOptimizerId } from "../optimizers";
import type { CnnTrainSnapshot, TrainRebuildReason, TrainSnapshot } from "./protocol";
import { TrainWorkerClient } from "./TrainWorkerClient";

/** Per-command argument shapes for the CNN train worker's `command` channel. */
export interface CnnCommandArgs {
  setDataset: { dataset: CnnConfig["dataset"] };
  setActivation: { activation: CnnActivationId };
  setLearningRate: { lr: number };
  setOptimizer: { optimizer: PlaygroundOptimizerId };
  setBatchSize: { bs: number };
  setRegularization: { regularization: CnnRegularizationId };
  setRegularizationRate: { rate: number };
  updateDataParams: { noise?: number; percTrainData?: number };
  /** No-arg command. */
  regenerateData: void;
  removeLayer: { index: number };
  setLayerFilters: { index: number; filters: number };
  setLayerKernelSize: { index: number; kernelSize: number };
  setLayerUnits: { index: number; units: number };
  setLayerPoolKind: { index: number; poolKind: "max" | "avg" };
  setInspectedExample: { index: number };
}

export interface CnnTrainClientOptions {
  /** Factory for the train worker (CNN WASM). Required unless `client` is given. */
  createWorker?: () => Worker;
  /** Inject an existing transport (used by tests). */
  client?: TrainWorkerClient;
  onTick: (snapshot: CnnTrainSnapshot) => void;
  onError?: (message: string) => void;
}

/** Tuple form so no-arg commands (e.g. `regenerateData`) are called without args. */
type CommandRest<K extends keyof CnnCommandArgs> = CnnCommandArgs[K] extends void
  ? []
  : [args: CnnCommandArgs[K]];

export class CnnTrainClient {
  private readonly client: TrainWorkerClient;
  private readonly onTick: (snapshot: CnnTrainSnapshot) => void;

  constructor(options: CnnTrainClientOptions) {
    this.onTick = options.onTick;
    const deliver = (s: TrainSnapshot) => {
      if (s.kind === "cnn") this.onTick(s);
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
      throw new Error("CnnTrainClient requires either createWorker or client");
    }
  }

  async init(config: CnnConfig): Promise<CnnTrainSnapshot> {
    const s = await this.client.init(config);
    return assertCnn(s);
  }

  rebuild(reason: "reset" | "resetWeights"): Promise<void>;
  rebuild(reason: "mode", payload: CnnMode): Promise<void>;
  rebuild(reason: "dataset", payload: CnnConfig["dataset"]): Promise<void>;
  rebuild(reason: "topology", payload: never): Promise<void>;
  rebuild(reason: TrainRebuildReason, payload?: unknown): Promise<void> {
    return this.client.rebuild(reason, payload);
  }

  command<K extends keyof CnnCommandArgs>(name: K, ...rest: CommandRest<K>): void {
    this.client.command(name, rest[0]);
  }

  inspect(exampleIndex?: number): void {
    this.client.inspect(exampleIndex);
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

  get snapshot(): CnnTrainSnapshot | null {
    const s = this.client.snapshot;
    return s && s.kind === "cnn" ? s : null;
  }
}

function assertCnn(s: TrainSnapshot): CnnTrainSnapshot {
  if (s.kind !== "cnn") {
    throw new Error(`CnnTrainClient expected a cnn snapshot, got "${s.kind}"`);
  }
  return s;
}
