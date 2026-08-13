/**
 * Type-safe wrapper around {@link TrainWorkerClient} for the network (MLP with a
 * computational graph) engine. See {@link CnnTrainClient} for the rationale:
 * the transport stays untyped, this wrapper restores per-engine types on the
 * main thread (concrete config, uniform rebuild payload, typed commands, and an
 * `onTick` narrowed to {@link NetworkTrainSnapshot}).
 */

import type { GraphSnapshot } from "../network/graph/types";
import type {
  NetworkActivationId,
  NetworkPlaygroundConfig,
  NetworkRegularizationId,
} from "../network/engine";
import type { PlaygroundOptimizerId } from "../optimizers";
import type {
  NetworkTrainSnapshot,
  TrainRebuildReason,
  TrainSnapshot,
} from "./protocol";
import { TrainWorkerClient } from "./TrainWorkerClient";

/** A labeled 2D point, as carried across the worker boundary. */
export interface NetworkDataPoint {
  x: number;
  y: number;
  label: number;
}

/**
 * Init / rebuild payload: the engine config plus the graph + data needed to
 * rebuild the worker-side display engine.
 */
export interface NetworkWorkerPayload {
  config: NetworkPlaygroundConfig;
  graphSnapshot?: GraphSnapshot;
  trainData?: NetworkDataPoint[];
  testData?: NetworkDataPoint[];
}

/** Per-command argument shapes for the network train worker's `command` channel. */
export interface NetworkCommandArgs {
  setLearningRate: { lr: number };
  setOptimizer: { optimizer: PlaygroundOptimizerId };
  setActivation: { activation: NetworkActivationId };
  setRegularization: { regularization: NetworkRegularizationId };
  setRegularizationRate: { rate: number };
  setBatchSize: { bs: number };
  syncGraph: {
    graphSnapshot: GraphSnapshot;
    trainData?: NetworkDataPoint[];
    testData?: NetworkDataPoint[];
  };
}

export interface NetworkTrainClientOptions {
  createWorker?: () => Worker;
  client?: TrainWorkerClient;
  onTick: (snapshot: NetworkTrainSnapshot) => void;
  onError?: (message: string) => void;
}

type CommandRest<K extends keyof NetworkCommandArgs> = NetworkCommandArgs[K] extends void
  ? []
  : [args: NetworkCommandArgs[K]];

export class NetworkTrainClient {
  private readonly client: TrainWorkerClient;
  private readonly onTick: (snapshot: NetworkTrainSnapshot) => void;

  constructor(options: NetworkTrainClientOptions) {
    this.onTick = options.onTick;
    const deliver = (s: TrainSnapshot) => {
      if (s.kind === "network") this.onTick(s);
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
      throw new Error("NetworkTrainClient requires either createWorker or client");
    }
  }

  async init(payload: NetworkWorkerPayload): Promise<NetworkTrainSnapshot> {
    const s = await this.client.init(payload);
    return assertNetwork(s);
  }

  rebuild(reason: TrainRebuildReason, payload: NetworkWorkerPayload): Promise<void> {
    return this.client.rebuild(reason, payload);
  }

  command<K extends keyof NetworkCommandArgs>(name: K, ...rest: CommandRest<K>): void {
    this.client.command(name, rest[0]);
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

  get snapshot(): NetworkTrainSnapshot | null {
    const s = this.client.snapshot;
    return s && s.kind === "network" ? s : null;
  }
}

function assertNetwork(s: TrainSnapshot): NetworkTrainSnapshot {
  if (s.kind !== "network") {
    throw new Error(`NetworkTrainClient expected a network snapshot, got "${s.kind}"`);
  }
  return s;
}
