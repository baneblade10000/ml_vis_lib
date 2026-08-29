/* tslint:disable */
/* eslint-disable */

export class WasmTransformerEngine {
    free(): void;
    [Symbol.dispose](): void;
    get_loss(): number;
    get_step(): bigint;
    constructor(config_json: string, task: string);
    reroll_sample(): any;
    reset(task: string): void;
    set_learning_rate(lr: number): void;
    /**
     * Full snapshot for the UI (greedy decode + attention + loss history).
     */
    snapshot(): any;
    /**
     * Runs `n` Adam steps; returns the EMA loss. No snapshot allocation —
     * this is the Play hot path.
     */
    train_steps(n: number): number;
    /**
     * Runs `n` steps, then returns a fresh snapshot.
     */
    train_steps_snapshot(n: number): any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmtransformerengine_free: (a: number, b: number) => void;
    readonly wasmtransformerengine_get_loss: (a: number) => number;
    readonly wasmtransformerengine_get_step: (a: number) => bigint;
    readonly wasmtransformerengine_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly wasmtransformerengine_reroll_sample: (a: number) => [number, number, number];
    readonly wasmtransformerengine_reset: (a: number, b: number, c: number) => [number, number];
    readonly wasmtransformerengine_set_learning_rate: (a: number, b: number) => void;
    readonly wasmtransformerengine_snapshot: (a: number) => [number, number, number];
    readonly wasmtransformerengine_train_steps: (a: number, b: number) => number;
    readonly wasmtransformerengine_train_steps_snapshot: (a: number, b: number) => [number, number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
