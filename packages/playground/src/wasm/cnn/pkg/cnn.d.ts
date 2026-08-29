/* tslint:disable */
/* eslint-disable */

export class WasmCnnEngine {
    free(): void;
    [Symbol.dispose](): void;
    accumulateBatch(indices: Uint32Array): number;
    addGradSums(grads: Float32Array): void;
    applyConfigJson(config_json: string): void;
    applySgd(lr: number, batch_n: number): void;
    backend(): string;
    bumpEpoch(loss_train: number): void;
    configJson(): string;
    /**
     * Prefer WebGPU; fall back to CPU if adapter/device unavailable.
     */
    static create(config_json: string): Promise<WasmCnnEngine>;
    epoch(): number;
    exportGrads(): Float32Array;
    exportParams(): Float32Array;
    featureMaps(include_kernels?: boolean | null): any;
    inspected(): number;
    loadParams(params: Float32Array): void;
    lossHistory(): any;
    metas(): any;
    nTrain(): number;
    /**
     * Sync CPU-only constructor (no WebGPU request).
     */
    constructor(config_json: string);
    paramCount(): number;
    predictGallery(max: number): any;
    probability(): number;
    pushLossHistory(): void;
    rebuildModel(): void;
    refreshAccuracy(): void;
    refreshMetrics(): void;
    setData1d(signals: Float32Array, labels: Int32Array, n: number, train_n: number): void;
    setData2d(images: Float32Array, labels: Int32Array, n: number, train_n: number): void;
    setInspected(index: number): void;
    stats(): any;
    /**
     * Play hot path: train only — no history push, no JSValue marshalling.
     */
    trainEpoch(): void;
    /**
     * Step / pause: train + append loss curve point.
     */
    trainEpochRecord(): any;
    zeroAllGrads(): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmcnnengine_free: (a: number, b: number) => void;
    readonly wasmcnnengine_accumulateBatch: (a: number, b: number, c: number) => number;
    readonly wasmcnnengine_addGradSums: (a: number, b: number, c: number) => void;
    readonly wasmcnnengine_applyConfigJson: (a: number, b: number, c: number) => [number, number];
    readonly wasmcnnengine_applySgd: (a: number, b: number, c: number) => void;
    readonly wasmcnnengine_backend: (a: number) => [number, number];
    readonly wasmcnnengine_bumpEpoch: (a: number, b: number) => void;
    readonly wasmcnnengine_configJson: (a: number) => [number, number, number, number];
    readonly wasmcnnengine_create: (a: number, b: number) => any;
    readonly wasmcnnengine_epoch: (a: number) => number;
    readonly wasmcnnengine_exportGrads: (a: number) => [number, number];
    readonly wasmcnnengine_exportParams: (a: number) => [number, number];
    readonly wasmcnnengine_featureMaps: (a: number, b: number) => [number, number, number];
    readonly wasmcnnengine_inspected: (a: number) => number;
    readonly wasmcnnengine_loadParams: (a: number, b: number, c: number) => void;
    readonly wasmcnnengine_lossHistory: (a: number) => [number, number, number];
    readonly wasmcnnengine_metas: (a: number) => [number, number, number];
    readonly wasmcnnengine_nTrain: (a: number) => number;
    readonly wasmcnnengine_new: (a: number, b: number) => [number, number, number];
    readonly wasmcnnengine_paramCount: (a: number) => number;
    readonly wasmcnnengine_predictGallery: (a: number, b: number) => [number, number, number];
    readonly wasmcnnengine_probability: (a: number) => number;
    readonly wasmcnnengine_pushLossHistory: (a: number) => void;
    readonly wasmcnnengine_rebuildModel: (a: number) => void;
    readonly wasmcnnengine_refreshAccuracy: (a: number) => void;
    readonly wasmcnnengine_refreshMetrics: (a: number) => void;
    readonly wasmcnnengine_setData1d: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly wasmcnnengine_setData2d: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly wasmcnnengine_setInspected: (a: number, b: number) => void;
    readonly wasmcnnengine_stats: (a: number) => [number, number, number];
    readonly wasmcnnengine_trainEpoch: (a: number) => void;
    readonly wasmcnnengine_trainEpochRecord: (a: number) => [number, number, number];
    readonly wasmcnnengine_zeroAllGrads: (a: number) => void;
    readonly wasm_bindgen_af390c650355dc71___convert__closures_____invoke___wasm_bindgen_af390c650355dc71___JsValue__core_9b3796e30d99ddb7___result__Result_____wasm_bindgen_af390c650355dc71___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_af390c650355dc71___convert__closures_____invoke___js_sys_5151b4f7262fa7a5___Function_fn_wasm_bindgen_af390c650355dc71___JsValue_____wasm_bindgen_af390c650355dc71___sys__Undefined___js_sys_5151b4f7262fa7a5___Function_fn_wasm_bindgen_af390c650355dc71___JsValue_____wasm_bindgen_af390c650355dc71___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
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
