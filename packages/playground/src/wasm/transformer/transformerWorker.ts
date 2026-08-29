/// <reference lib="webworker" />
/**
 * Transformer train worker — Rust WASM owns forward/backward/Adam/decode.
 * The main thread only renders snapshots; the hot Play path trains N steps
 * per tick without any snapshot allocation.
 */
import { isTransformerTaskId, type TransformerSnapshot, type TransformerTaskId } from "@ml-vis/core/transformer";
import init, { WasmTransformerEngine } from "./pkg/transformer.js";
import wasmUrl from "./pkg/transformer_bg.wasm?url";

declare const self: DedicatedWorkerGlobalScope;

export type ToTransformerWorker =
  | { type: "init"; task: TransformerTaskId }
  | { type: "play"; stepsPerTick: number }
  | { type: "pause" }
  | { type: "stepOnce" }
  | { type: "reset"; task: TransformerTaskId }
  | { type: "setTask"; task: TransformerTaskId }
  | { type: "setLearningRate"; lr: number }
  | { type: "rerollSample" };

export type FromTransformerWorker =
  | { type: "ready"; backend: string }
  | { type: "snapshot"; snapshot: TransformerSnapshot }
  | { type: "error"; message: string };

let engine: WasmTransformerEngine | null = null;
let wasmReady: Promise<void> | null = null;
let playTimer: ReturnType<typeof setTimeout> | null = null;
let playing = false;
let pendingTask: TransformerTaskId = "reverse";

function post(msg: FromTransformerWorker): void {
  self.postMessage(msg);
}

async function ensureWasm(): Promise<void> {
  if (!wasmReady) {
    wasmReady = init({ module_or_path: wasmUrl }).then(() => undefined);
  }
  await wasmReady;
}

function postSnapshot(): void {
  if (!engine) return;
  try {
    post({ type: "snapshot", snapshot: engine.snapshot() as unknown as TransformerSnapshot });
  } catch (e) {
    post({ type: "error", message: String(e) });
  }
}

function stopPlay(): void {
  playing = false;
  if (playTimer !== null) {
    clearTimeout(playTimer);
    playTimer = null;
  }
}

/** Fixed-rhythm train loop: trains N steps and posts a fresh snapshot. */
function scheduleTick(stepsPerTick: number): void {
  playTimer = setTimeout(() => {
    if (!playing || !engine) return;
    try {
      post({
        type: "snapshot",
        snapshot: engine.train_steps_snapshot(stepsPerTick) as unknown as TransformerSnapshot,
      });
    } catch (e) {
      post({ type: "error", message: String(e) });
      stopPlay();
      return;
    }
    scheduleTick(stepsPerTick);
  }, 50);
}

function handleMessage(msg: ToTransformerWorker): void {
  void (async () => {
    try {
      await ensureWasm();
      // The engine is constructed lazily on the first message, so `init`
      // just records the task before that construction happens.
      if (msg.type === "init" && isTransformerTaskId(msg.task)) pendingTask = msg.task;
      if (!engine) {
        engine = new WasmTransformerEngine("", pendingTask);
        post({ type: "ready", backend: "rust-wasm" });
        postSnapshot();
      }
      switch (msg.type) {
        case "init":
          break;
        case "play":
          if (!playing) {
            playing = true;
            scheduleTick(msg.stepsPerTick);
          }
          break;
        case "pause":
          stopPlay();
          postSnapshot();
          break;
        case "stepOnce":
          stopPlay();
          post({
            type: "snapshot",
            snapshot: engine.train_steps_snapshot(1) as unknown as TransformerSnapshot,
          });
          break;
        case "reset":
          stopPlay();
          engine.reset(msg.task);
          postSnapshot();
          break;
        case "setTask":
          if (!isTransformerTaskId(msg.task)) {
            post({ type: "error", message: `unknown task: ${msg.task}` });
            return;
          }
          stopPlay();
          engine.reset(msg.task);
          postSnapshot();
          break;
        case "setLearningRate":
          engine.set_learning_rate(msg.lr);
          break;
        case "rerollSample":
          engine.reroll_sample();
          postSnapshot();
          break;
      }
    } catch (e) {
      post({ type: "error", message: String(e) });
    }
  })();
}

self.addEventListener("message", (event: MessageEvent<ToTransformerWorker>) => {
  handleMessage(event.data);
});
