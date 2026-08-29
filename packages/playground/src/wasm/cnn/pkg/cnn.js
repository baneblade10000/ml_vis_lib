/* @ts-self-types="./cnn.d.ts" */

export class WasmCnnEngine {
    static __wrap(ptr) {
        const obj = Object.create(WasmCnnEngine.prototype);
        obj.__wbg_ptr = ptr;
        WasmCnnEngineFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmCnnEngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmcnnengine_free(ptr, 0);
    }
    /**
     * @param {Uint32Array} indices
     * @returns {number}
     */
    accumulateBatch(indices) {
        const ptr0 = passArray32ToWasm0(indices, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmcnnengine_accumulateBatch(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * @param {Float32Array} grads
     */
    addGradSums(grads) {
        const ptr0 = passArrayF32ToWasm0(grads, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmcnnengine_addGradSums(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {string} config_json
     */
    applyConfigJson(config_json) {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmcnnengine_applyConfigJson(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {number} lr
     * @param {number} batch_n
     */
    applySgd(lr, batch_n) {
        wasm.wasmcnnengine_applySgd(this.__wbg_ptr, lr, batch_n);
    }
    /**
     * @returns {string}
     */
    backend() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmcnnengine_backend(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {number} loss_train
     */
    bumpEpoch(loss_train) {
        wasm.wasmcnnengine_bumpEpoch(this.__wbg_ptr, loss_train);
    }
    /**
     * @returns {string}
     */
    configJson() {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmcnnengine_configJson(this.__wbg_ptr);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Prefer WebGPU; fall back to CPU if adapter/device unavailable.
     * @param {string} config_json
     * @returns {Promise<WasmCnnEngine>}
     */
    static create(config_json) {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmcnnengine_create(ptr0, len0);
        return ret;
    }
    /**
     * @returns {number}
     */
    epoch() {
        const ret = wasm.wasmcnnengine_epoch(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Float32Array}
     */
    exportGrads() {
        const ret = wasm.wasmcnnengine_exportGrads(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Float32Array}
     */
    exportParams() {
        const ret = wasm.wasmcnnengine_exportParams(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {boolean | null} [include_kernels]
     * @returns {any}
     */
    featureMaps(include_kernels) {
        const ret = wasm.wasmcnnengine_featureMaps(this.__wbg_ptr, isLikeNone(include_kernels) ? 0xFFFFFF : include_kernels ? 1 : 0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {number}
     */
    inspected() {
        const ret = wasm.wasmcnnengine_inspected(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {Float32Array} params
     */
    loadParams(params) {
        const ptr0 = passArrayF32ToWasm0(params, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmcnnengine_loadParams(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @returns {any}
     */
    lossHistory() {
        const ret = wasm.wasmcnnengine_lossHistory(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {any}
     */
    metas() {
        const ret = wasm.wasmcnnengine_metas(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {number}
     */
    nTrain() {
        const ret = wasm.wasmcnnengine_nTrain(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Sync CPU-only constructor (no WebGPU request).
     * @param {string} config_json
     */
    constructor(config_json) {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmcnnengine_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WasmCnnEngineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    paramCount() {
        const ret = wasm.wasmcnnengine_paramCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} max
     * @returns {any}
     */
    predictGallery(max) {
        const ret = wasm.wasmcnnengine_predictGallery(this.__wbg_ptr, max);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * @returns {number}
     */
    probability() {
        const ret = wasm.wasmcnnengine_probability(this.__wbg_ptr);
        return ret;
    }
    pushLossHistory() {
        wasm.wasmcnnengine_pushLossHistory(this.__wbg_ptr);
    }
    rebuildModel() {
        wasm.wasmcnnengine_rebuildModel(this.__wbg_ptr);
    }
    refreshAccuracy() {
        wasm.wasmcnnengine_refreshAccuracy(this.__wbg_ptr);
    }
    refreshMetrics() {
        wasm.wasmcnnengine_refreshMetrics(this.__wbg_ptr);
    }
    /**
     * @param {Float32Array} signals
     * @param {Int32Array} labels
     * @param {number} n
     * @param {number} train_n
     */
    setData1d(signals, labels, n, train_n) {
        const ptr0 = passArrayF32ToWasm0(signals, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(labels, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmcnnengine_setData1d(this.__wbg_ptr, ptr0, len0, ptr1, len1, n, train_n);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {Float32Array} images
     * @param {Int32Array} labels
     * @param {number} n
     * @param {number} train_n
     */
    setData2d(images, labels, n, train_n) {
        const ptr0 = passArrayF32ToWasm0(images, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(labels, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmcnnengine_setData2d(this.__wbg_ptr, ptr0, len0, ptr1, len1, n, train_n);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {number} index
     */
    setInspected(index) {
        wasm.wasmcnnengine_setInspected(this.__wbg_ptr, index);
    }
    /**
     * @returns {any}
     */
    stats() {
        const ret = wasm.wasmcnnengine_stats(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Play hot path: train only — no history push, no JSValue marshalling.
     */
    trainEpoch() {
        wasm.wasmcnnengine_trainEpoch(this.__wbg_ptr);
    }
    /**
     * Step / pause: train + append loss curve point.
     * @returns {any}
     */
    trainEpochRecord() {
        const ret = wasm.wasmcnnengine_trainEpochRecord(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    zeroAllGrads() {
        wasm.wasmcnnengine_zeroAllGrads(this.__wbg_ptr);
    }
}
if (Symbol.dispose) WasmCnnEngine.prototype[Symbol.dispose] = WasmCnnEngine.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_408e67f47ca7b58b: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_String_8564e559799eccda: function(arg0, arg1) {
            const ret = String(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_Window_412fe051c1aa1519: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_349300f9b277afe1: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
        },
        __wbg___wbindgen_debug_string_a57024b9c6e4a48b: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_5e4570eb24ffa122: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_7d13f41e1a2d5140: function(arg0) {
            const ret = arg0 === null;
            return ret;
        },
        __wbg___wbindgen_is_undefined_6cff064c44e0d823: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_throw_bb96b2010945f0bc: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_be22cc64ae6946a0: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_call_35dba3c747ad7521: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_error_757e9472f8410341: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_gpu_bafbc1407fe850fb: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_instanceof_GpuAdapter_aff4b0f95a6c1c3e: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUAdapter;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_limits_805a5a24eacdef89: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_maxBindGroups_aeb19ade452446c6: function(arg0) {
            const ret = arg0.maxBindGroups;
            return ret;
        },
        __wbg_maxBindingsPerBindGroup_56b9cd783b976459: function(arg0) {
            const ret = arg0.maxBindingsPerBindGroup;
            return ret;
        },
        __wbg_maxBufferSize_b9cfa105ccd49524: function(arg0) {
            const ret = arg0.maxBufferSize;
            return ret;
        },
        __wbg_maxColorAttachmentBytesPerSample_3853759407ab3c40: function(arg0) {
            const ret = arg0.maxColorAttachmentBytesPerSample;
            return ret;
        },
        __wbg_maxColorAttachments_a9a3c5bc728fb56f: function(arg0) {
            const ret = arg0.maxColorAttachments;
            return ret;
        },
        __wbg_maxComputeInvocationsPerWorkgroup_732f87215035d9e5: function(arg0) {
            const ret = arg0.maxComputeInvocationsPerWorkgroup;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeX_4f1d6552edeba82a: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeX;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeY_170c377843fcdda9: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeY;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeZ_78bd13bc9226bc99: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeZ;
            return ret;
        },
        __wbg_maxComputeWorkgroupStorageSize_0a6873ffe86d432d: function(arg0) {
            const ret = arg0.maxComputeWorkgroupStorageSize;
            return ret;
        },
        __wbg_maxComputeWorkgroupsPerDimension_7f64fa252d98d9e0: function(arg0) {
            const ret = arg0.maxComputeWorkgroupsPerDimension;
            return ret;
        },
        __wbg_maxDynamicStorageBuffersPerPipelineLayout_548c4d8427692343: function(arg0) {
            const ret = arg0.maxDynamicStorageBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxDynamicUniformBuffersPerPipelineLayout_575f81e5a619fda4: function(arg0) {
            const ret = arg0.maxDynamicUniformBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxSampledTexturesPerShaderStage_4b0d0d0deb7f9173: function(arg0) {
            const ret = arg0.maxSampledTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxSamplersPerShaderStage_122e1c314b7d5f0e: function(arg0) {
            const ret = arg0.maxSamplersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageBufferBindingSize_69368e8c4a720d65: function(arg0) {
            const ret = arg0.maxStorageBufferBindingSize;
            return ret;
        },
        __wbg_maxStorageBuffersPerShaderStage_483da9a48e09b2cd: function(arg0) {
            const ret = arg0.maxStorageBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageTexturesPerShaderStage_825095cb824c2a90: function(arg0) {
            const ret = arg0.maxStorageTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxTextureArrayLayers_311d9cd973092ad3: function(arg0) {
            const ret = arg0.maxTextureArrayLayers;
            return ret;
        },
        __wbg_maxTextureDimension1D_f14696527b4dd4c9: function(arg0) {
            const ret = arg0.maxTextureDimension1D;
            return ret;
        },
        __wbg_maxTextureDimension2D_8a888981a9a496a3: function(arg0) {
            const ret = arg0.maxTextureDimension2D;
            return ret;
        },
        __wbg_maxTextureDimension3D_af7a8c47b3a93760: function(arg0) {
            const ret = arg0.maxTextureDimension3D;
            return ret;
        },
        __wbg_maxUniformBufferBindingSize_5cab04d98886e7d3: function(arg0) {
            const ret = arg0.maxUniformBufferBindingSize;
            return ret;
        },
        __wbg_maxUniformBuffersPerShaderStage_14f4345c06c80500: function(arg0) {
            const ret = arg0.maxUniformBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxVertexAttributes_73ce901689262af1: function(arg0) {
            const ret = arg0.maxVertexAttributes;
            return ret;
        },
        __wbg_maxVertexBufferArrayStride_4cb22981054e6df0: function(arg0) {
            const ret = arg0.maxVertexBufferArrayStride;
            return ret;
        },
        __wbg_maxVertexBuffers_a6df2bc183ca7af0: function(arg0) {
            const ret = arg0.maxVertexBuffers;
            return ret;
        },
        __wbg_minStorageBufferOffsetAlignment_2039bfddd5b42bd9: function(arg0) {
            const ret = arg0.minStorageBufferOffsetAlignment;
            return ret;
        },
        __wbg_minUniformBufferOffsetAlignment_4366c5e24c3a2e2f: function(arg0) {
            const ret = arg0.minUniformBufferOffsetAlignment;
            return ret;
        },
        __wbg_navigator_6cfdd5fa246d910f: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_navigator_e5c345298a9609cd: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_new_116be93542d39019: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_ebe3e0f6837f0879: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_typed_cceaf62d8d95e9f2: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen_af390c650355dc71___convert__closures_____invoke___js_sys_5151b4f7262fa7a5___Function_fn_wasm_bindgen_af390c650355dc71___JsValue_____wasm_bindgen_af390c650355dc71___sys__Undefined___js_sys_5151b4f7262fa7a5___Function_fn_wasm_bindgen_af390c650355dc71___JsValue_____wasm_bindgen_af390c650355dc71___sys__Undefined_______true_(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_parse_1cc93481b0865939: function() { return handleError(function (arg0, arg1) {
            const ret = JSON.parse(getStringFromWasm0(arg0, arg1));
            return ret;
        }, arguments); },
        __wbg_push_adb0107829f02d75: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_queueMicrotask_ac694eae12e92dfb: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_queueMicrotask_be5fe34a8f4cad4d: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queue_3e40156d83b9183e: function(arg0) {
            const ret = arg0.queue;
            return ret;
        },
        __wbg_requestAdapter_245da40985c2fdc5: function(arg0, arg1) {
            const ret = arg0.requestAdapter(arg1);
            return ret;
        },
        __wbg_requestDevice_28434913a23418c4: function(arg0, arg1) {
            const ret = arg0.requestDevice(arg1);
            return ret;
        },
        __wbg_resolve_020f95d838c6ef25: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_set_6be42768c690e380: function(arg0, arg1, arg2) {
            arg0[arg1] = arg2;
        },
        __wbg_set_8155bb79a948541b: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_a80955eb93b145c6: function(arg0, arg1, arg2) {
            arg0[arg1 >>> 0] = arg2;
        },
        __wbg_set_label_3e06143ad04772ae: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_power_preference_915480f4b9565dc2: function(arg0, arg1) {
            arg0.powerPreference = __wbindgen_enum_GpuPowerPreference[arg1];
        },
        __wbg_set_required_features_42347bf311233eb6: function(arg0, arg1) {
            arg0.requiredFeatures = arg1;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_THIS_466428f93b4eaa76: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_c7aea38d4de089bc: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_42d4fae05e59267a: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_e0db14a0eba6a812: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_then_7026b513a94278a8: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_then_72819b8d4e081fb5: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_wasmcnnengine_new: function(arg0) {
            const ret = WasmCnnEngine.__wrap(arg0);
            return ret;
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 69, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_af390c650355dc71___convert__closures_____invoke___wasm_bindgen_af390c650355dc71___JsValue__core_9b3796e30d99ddb7___result__Result_____wasm_bindgen_af390c650355dc71___JsError___true_);
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000004: function(arg0) {
            // Cast intrinsic for `U64 -> Externref`.
            const ret = BigInt.asUintN(64, arg0);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./cnn_bg.js": import0,
    };
}

function wasm_bindgen_af390c650355dc71___convert__closures_____invoke___wasm_bindgen_af390c650355dc71___JsValue__core_9b3796e30d99ddb7___result__Result_____wasm_bindgen_af390c650355dc71___JsError___true_(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_af390c650355dc71___convert__closures_____invoke___wasm_bindgen_af390c650355dc71___JsValue__core_9b3796e30d99ddb7___result__Result_____wasm_bindgen_af390c650355dc71___JsError___true_(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_af390c650355dc71___convert__closures_____invoke___js_sys_5151b4f7262fa7a5___Function_fn_wasm_bindgen_af390c650355dc71___JsValue_____wasm_bindgen_af390c650355dc71___sys__Undefined___js_sys_5151b4f7262fa7a5___Function_fn_wasm_bindgen_af390c650355dc71___JsValue_____wasm_bindgen_af390c650355dc71___sys__Undefined_______true_(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen_af390c650355dc71___convert__closures_____invoke___js_sys_5151b4f7262fa7a5___Function_fn_wasm_bindgen_af390c650355dc71___JsValue_____wasm_bindgen_af390c650355dc71___sys__Undefined___js_sys_5151b4f7262fa7a5___Function_fn_wasm_bindgen_af390c650355dc71___JsValue_____wasm_bindgen_af390c650355dc71___sys__Undefined_______true_(arg0, arg1, arg2, arg3);
}


const __wbindgen_enum_GpuPowerPreference = ["low-power", "high-performance"];
const WasmCnnEngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmcnnengine_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('cnn_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
