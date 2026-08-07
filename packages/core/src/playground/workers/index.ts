export type {
  CnnLayerView,
  CnnTrainSnapshot,
  FromTrainWorker,
  MlpTrainSnapshot,
  NetworkTrainSnapshot,
  ToTrainWorker,
  TrainRebuildReason,
  TrainSnapshot,
} from "./protocol";
export { TrainWorkerClient, canUseTrainWorkers } from "./TrainWorkerClient";
export type { TrainWorkerClientOptions, TrainWorkerErrorHandler, TrainWorkerTickHandler } from "./TrainWorkerClient";
export { defaultShardCount } from "./shardCount";
// create*TrainWorker are NOT re-exported here — import from
// `@ml-vis/core/workers/createWorkers` so bundlers keep worker URLs intact.
