let wasm;

let heap = new Array(128).fill(undefined);

heap.push(undefined, null, true, false);

function getObject(idx) { return heap[idx]; }

function isLikeNone(x) {
    return x === undefined || x === null;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
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

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let heap_next = heap.length;

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export(addHeapObject(e));
    }
}

let cachedFloat64ArrayMemory0 = null;

function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

function dropObject(idx) {
    if (idx < 132) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
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

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => state.dtor(state.a, state.b));

function makeMutClosure(arg0, arg1, dtor, f) {
    const state = { a: arg0, b: arg1, cnt: 1, dtor };
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
            state.dtor(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

export function main() {
    wasm.main();
}

/**
 * @param {Array<any>} points
 * @returns {JsDataset}
 */
export function generate_dataset_from_points(points) {
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.generate_dataset_from_points(retptr, addHeapObject(points));
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
        if (r2) {
            throw takeObject(r1);
        }
        return JsDataset.__wrap(r0);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
    }
}

function __wasm_bindgen_func_elem_830(arg0, arg1, arg2) {
    wasm.__wasm_bindgen_func_elem_830(arg0, arg1, addHeapObject(arg2));
}

function __wasm_bindgen_func_elem_1210(arg0, arg1, arg2, arg3) {
    wasm.__wasm_bindgen_func_elem_1210(arg0, arg1, addHeapObject(arg2), addHeapObject(arg3));
}

/**
 * @enum {0 | 1 | 2 | 3 | 4}
 */
export const JsOptimizerType = Object.freeze({
    GD: 0, "0": "GD",
    SGD: 1, "1": "SGD",
    SGDMomentum: 2, "2": "SGDMomentum",
    RMSProp: 3, "3": "RMSProp",
    Adam: 4, "4": "Adam",
});
/**
 * @enum {0 | 1 | 2 | 3}
 */
export const JsRegularizationType = Object.freeze({
    None: 0, "0": "None",
    L1: 1, "1": "L1",
    L2: 2, "2": "L2",
    ElasticNet: 3, "3": "ElasticNet",
});
/**
 * @enum {0 | 1 | 2}
 */
export const JsTaskType = Object.freeze({
    BinaryClassification: 0, "0": "BinaryClassification",
    MultiClassification: 1, "1": "MultiClassification",
    Regression: 2, "2": "Regression",
});

const DataConverterFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_dataconverter_free(ptr >>> 0, 1));
/**
 * Data conversion utilities between JavaScript and Rust types
 */
export class DataConverter {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DataConverterFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_dataconverter_free(ptr, 0);
    }
    /**
     * Convert JavaScript 2D array to JsTensor
     * @param {Array<any>} array
     * @returns {JsTensor}
     */
    static array_to_tensor(array) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.dataconverter_array_to_tensor(retptr, addHeapObject(array));
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Convert JsTensor to JavaScript 2D array
     * @param {JsTensor} tensor
     * @returns {Array<any>}
     */
    static tensor_to_array(tensor) {
        _assertClass(tensor, JsTensor);
        const ret = wasm.dataconverter_tensor_to_array(tensor.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * Convert flat JavaScript array to JsTensor with specified shape
     * @param {Float64Array} flat_array
     * @param {number} rows
     * @param {number} cols
     * @returns {JsTensor}
     */
    static flat_array_to_tensor(flat_array, rows, cols) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.dataconverter_flat_array_to_tensor(retptr, addHeapObject(flat_array), rows, cols);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Convert JsTensor to flat JavaScript array
     * @param {JsTensor} tensor
     * @returns {Float64Array}
     */
    static tensor_to_flat_array(tensor) {
        _assertClass(tensor, JsTensor);
        const ret = wasm.dataconverter_tensor_to_flat_array(tensor.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * Create tensor from CSV-like string data
     * @param {string} csv_string
     * @param {boolean} has_header
     * @param {string} delimiter
     * @returns {JsTensor}
     */
    static csv_to_tensor(csv_string, has_header, delimiter) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(csv_string, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passStringToWasm0(delimiter, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
            const len1 = WASM_VECTOR_LEN;
            wasm.dataconverter_csv_to_tensor(retptr, ptr0, len0, has_header, ptr1, len1);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Convert tensor to CSV-like string
     * @param {JsTensor} tensor
     * @param {string} delimiter
     * @returns {string}
     */
    static tensor_to_csv(tensor, delimiter) {
        let deferred2_0;
        let deferred2_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(tensor, JsTensor);
            const ptr0 = passStringToWasm0(delimiter, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
            const len0 = WASM_VECTOR_LEN;
            wasm.dataconverter_tensor_to_csv(retptr, tensor.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred2_0 = r0;
            deferred2_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Normalize tensor values to [0, 1] range
     * @param {JsTensor} tensor
     * @returns {JsTensor}
     */
    static normalize_min_max(tensor) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(tensor, JsTensor);
            wasm.dataconverter_normalize_min_max(retptr, tensor.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Standardize tensor values (z-score normalization)
     * @param {JsTensor} tensor
     * @returns {JsTensor}
     */
    static standardize(tensor) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(tensor, JsTensor);
            wasm.dataconverter_standardize(retptr, tensor.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) DataConverter.prototype[Symbol.dispose] = DataConverter.prototype.free;

const JsDataPointFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jsdatapoint_free(ptr >>> 0, 1));

export class JsDataPoint {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsDataPointFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jsdatapoint_free(ptr, 0);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} label
     */
    constructor(x, y, label) {
        const ret = wasm.jsdatapoint_new(x, y, label);
        this.__wbg_ptr = ret >>> 0;
        JsDataPointFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get x() {
        const ret = wasm.jsdatapoint_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get y() {
        const ret = wasm.jsdatapoint_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get label() {
        const ret = wasm.jsdatapoint_label(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) JsDataPoint.prototype[Symbol.dispose] = JsDataPoint.prototype.free;

const JsDatasetFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jsdataset_free(ptr >>> 0, 1));
/**
 * Dataset wrapper for JavaScript
 */
export class JsDataset {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(JsDataset.prototype);
        obj.__wbg_ptr = ptr;
        JsDatasetFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsDatasetFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jsdataset_free(ptr, 0);
    }
    /**
     * Create a new dataset from features and labels (defaulting to binary classification)
     * @param {Array<any>} features
     * @param {Float64Array} labels
     */
    constructor(features, labels) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsdataset_new(retptr, addHeapObject(features), addHeapObject(labels));
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            JsDatasetFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {JsTaskType}
     */
    get task_type() {
        const ret = wasm.jsdataset_task_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the number of samples
     * @returns {number}
     */
    len() {
        const ret = wasm.jsdataset_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the number of features per sample
     * @returns {number}
     */
    feature_count() {
        const ret = wasm.jsdataset_feature_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get features as a tensor
     * @returns {JsTensor}
     */
    features_tensor() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsdataset_features_tensor(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Get labels as a tensor
     * @returns {JsTensor}
     */
    labels_tensor() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsdataset_labels_tensor(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) JsDataset.prototype[Symbol.dispose] = JsDataset.prototype.free;

const JsMetricsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jsmetrics_free(ptr >>> 0, 1));

export class JsMetrics {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(JsMetrics.prototype);
        obj.__wbg_ptr = ptr;
        JsMetricsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsMetricsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jsmetrics_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get accuracy() {
        const ret = wasm.jsmetrics_accuracy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get loss() {
        const ret = wasm.jsmetrics_loss(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number | undefined}
     */
    get precision() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsmetrics_precision(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r2 = getDataViewMemory0().getFloat64(retptr + 8 * 1, true);
            return r0 === 0 ? undefined : r2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {number | undefined}
     */
    get recall() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsmetrics_recall(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r2 = getDataViewMemory0().getFloat64(retptr + 8 * 1, true);
            return r0 === 0 ? undefined : r2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {number | undefined}
     */
    get f1_score() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsmetrics_f1_score(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r2 = getDataViewMemory0().getFloat64(retptr + 8 * 1, true);
            return r0 === 0 ? undefined : r2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {number | undefined}
     */
    get mse() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsmetrics_mse(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r2 = getDataViewMemory0().getFloat64(retptr + 8 * 1, true);
            return r0 === 0 ? undefined : r2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) JsMetrics.prototype[Symbol.dispose] = JsMetrics.prototype.free;

const JsModelFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jsmodel_free(ptr >>> 0, 1));
/**
 * JavaScript-compatible wrapper for Neural Network Model
 */
export class JsModel {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(JsModel.prototype);
        obj.__wbg_ptr = ptr;
        JsModelFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsModelFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jsmodel_free(ptr, 0);
    }
    /**
     * Create a new empty model
     */
    constructor() {
        const ret = wasm.jsmodel_new();
        this.__wbg_ptr = ret >>> 0;
        JsModelFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Add a dense layer with ReLU activation
     * @param {number} input_size
     * @param {number} output_size
     */
    add_dense_relu(input_size, output_size) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsmodel_add_dense_relu(retptr, this.__wbg_ptr, input_size, output_size);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Add a dense layer with sigmoid activation
     * @param {number} input_size
     * @param {number} output_size
     */
    add_dense_sigmoid(input_size, output_size) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsmodel_add_dense_sigmoid(retptr, this.__wbg_ptr, input_size, output_size);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Add a dense layer with softmax activation (typically for output layer)
     * @param {number} input_size
     * @param {number} output_size
     */
    add_dense_softmax(input_size, output_size) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsmodel_add_dense_softmax(retptr, this.__wbg_ptr, input_size, output_size);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Forward pass through the model
     * @param {JsTensor} input
     * @returns {JsTensor}
     */
    forward(input) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(input, JsTensor);
            wasm.jsmodel_forward(retptr, this.__wbg_ptr, input.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Get model summary as a string
     * @returns {string}
     */
    summary() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsmodel_summary(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get total number of parameters
     * @returns {number}
     */
    param_count() {
        const ret = wasm.jsmodel_param_count(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) JsModel.prototype[Symbol.dispose] = JsModel.prototype.free;

const JsModelConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jsmodelconfig_free(ptr >>> 0, 1));

export class JsModelConfig {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsModelConfigFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jsmodelconfig_free(ptr, 0);
    }
    /**
     * @param {Array<any>} layers
     * @param {string} activation_fn
     * @param {JsTaskType} task_type
     */
    constructor(layers, activation_fn, task_type) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(activation_fn, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
            const len0 = WASM_VECTOR_LEN;
            wasm.jsmodelconfig_new(retptr, addHeapObject(layers), ptr0, len0, task_type);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            JsModelConfigFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {JsTaskType}
     */
    get task_type() {
        const ret = wasm.jsmodelconfig_task_type(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) JsModelConfig.prototype[Symbol.dispose] = JsModelConfig.prototype.free;

const JsOptimizerConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jsoptimizerconfig_free(ptr >>> 0, 1));

export class JsOptimizerConfig {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsOptimizerConfigFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jsoptimizerconfig_free(ptr, 0);
    }
    /**
     * @param {JsOptimizerType} optimizer_type
     * @param {number} learning_rate
     */
    constructor(optimizer_type, learning_rate) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsoptimizerconfig_new(retptr, optimizer_type, learning_rate);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            JsOptimizerConfigFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {JsOptimizerType}
     */
    get optimizer_type() {
        const ret = wasm.jsoptimizerconfig_optimizer_type(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get learning_rate() {
        const ret = wasm.jsoptimizerconfig_learning_rate(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) JsOptimizerConfig.prototype[Symbol.dispose] = JsOptimizerConfig.prototype.free;

const JsRegularizationConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jsregularizationconfig_free(ptr >>> 0, 1));

export class JsRegularizationConfig {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsRegularizationConfigFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jsregularizationconfig_free(ptr, 0);
    }
    /**
     * @param {JsRegularizationType} reg_type
     * @param {number} l1_lambda
     * @param {number} l2_lambda
     */
    constructor(reg_type, l1_lambda, l2_lambda) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jsregularizationconfig_new(retptr, reg_type, l1_lambda, l2_lambda);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            JsRegularizationConfigFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) JsRegularizationConfig.prototype[Symbol.dispose] = JsRegularizationConfig.prototype.free;

const JsTensorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jstensor_free(ptr >>> 0, 1));
/**
 * JavaScript-compatible wrapper for Tensor
 */
export class JsTensor {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(JsTensor.prototype);
        obj.__wbg_ptr = ptr;
        JsTensorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsTensorFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jstensor_free(ptr, 0);
    }
    /**
     * Create a new tensor from a JavaScript Float64Array
     *
     * # Arguments
     * * `data` - Flattened tensor data as Float64Array
     * * `rows` - Number of rows
     * * `cols` - Number of columns
     * @param {Float64Array} data
     * @param {number} rows
     * @param {number} cols
     */
    constructor(data, rows, cols) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.dataconverter_flat_array_to_tensor(retptr, addHeapObject(data), rows, cols);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            JsTensorFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Create a tensor filled with zeros
     * @param {number} rows
     * @param {number} cols
     * @returns {JsTensor}
     */
    static zeros(rows, cols) {
        const ret = wasm.jstensor_zeros(rows, cols);
        return JsTensor.__wrap(ret);
    }
    /**
     * Create a tensor filled with ones
     * @param {number} rows
     * @param {number} cols
     * @returns {JsTensor}
     */
    static ones(rows, cols) {
        const ret = wasm.jstensor_ones(rows, cols);
        return JsTensor.__wrap(ret);
    }
    /**
     * Create a tensor with random values between -1 and 1
     * @param {number} rows
     * @param {number} cols
     * @returns {JsTensor}
     */
    static random(rows, cols) {
        const ret = wasm.jstensor_random(rows, cols);
        return JsTensor.__wrap(ret);
    }
    /**
     * Get the shape of the tensor as [rows, cols]
     * @returns {Array<any>}
     */
    shape() {
        const ret = wasm.jstensor_shape(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * Get the tensor data as a flattened Float64Array
     * @returns {Float64Array}
     */
    data() {
        const ret = wasm.dataconverter_tensor_to_flat_array(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * Set whether this tensor requires gradients
     * @param {boolean} requires_grad
     */
    set_requires_grad(requires_grad) {
        wasm.jstensor_set_requires_grad(this.__wbg_ptr, requires_grad);
    }
    /**
     * Check if this tensor requires gradients
     * @returns {boolean}
     */
    requires_grad() {
        const ret = wasm.jstensor_requires_grad(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Get the gradient as a JsTensor (if available)
     * @returns {JsTensor | undefined}
     */
    gradient() {
        const ret = wasm.jstensor_gradient(this.__wbg_ptr);
        return ret === 0 ? undefined : JsTensor.__wrap(ret);
    }
    /**
     * Zero out the gradients
     */
    zero_grad() {
        wasm.jstensor_zero_grad(this.__wbg_ptr);
    }
    /**
     * Perform matrix multiplication
     * @param {JsTensor} other
     * @returns {JsTensor}
     */
    matmul(other) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(other, JsTensor);
            wasm.jstensor_matmul(retptr, this.__wbg_ptr, other.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Add two tensors
     * @param {JsTensor} other
     * @returns {JsTensor}
     */
    add(other) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(other, JsTensor);
            wasm.jstensor_add(retptr, this.__wbg_ptr, other.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Subtract two tensors
     * @param {JsTensor} other
     * @returns {JsTensor}
     */
    sub(other) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(other, JsTensor);
            wasm.jstensor_sub(retptr, this.__wbg_ptr, other.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Element-wise multiplication
     * @param {JsTensor} other
     * @returns {JsTensor}
     */
    mul(other) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(other, JsTensor);
            wasm.jstensor_mul(retptr, this.__wbg_ptr, other.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Scalar multiplication
     * @param {number} scalar
     * @returns {JsTensor}
     */
    mul_scalar(scalar) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jstensor_mul_scalar(retptr, this.__wbg_ptr, scalar);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Apply sigmoid activation
     * @returns {JsTensor}
     */
    sigmoid() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jstensor_sigmoid(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Apply ReLU activation
     * @returns {JsTensor}
     */
    relu() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jstensor_relu(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Apply tanh activation
     * @returns {JsTensor}
     */
    tanh() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jstensor_tanh(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Apply softmax activation
     * @returns {JsTensor}
     */
    softmax() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jstensor_softmax(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Compute mean of all elements
     * @returns {JsTensor}
     */
    mean() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jstensor_mean(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Perform backward pass (compute gradients)
     */
    backward() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jstensor_backward(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Clone the tensor
     * @returns {JsTensor}
     */
    clone() {
        const ret = wasm.jstensor_clone(this.__wbg_ptr);
        return JsTensor.__wrap(ret);
    }
    /**
     * Get a string representation of the tensor
     * @returns {string}
     */
    to_string() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jstensor_to_string(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Log the tensor to browser console (for debugging)
     */
    log() {
        wasm.jstensor_log(this.__wbg_ptr);
    }
}
if (Symbol.dispose) JsTensor.prototype[Symbol.dispose] = JsTensor.prototype.free;

const JsTrainerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jstrainer_free(ptr >>> 0, 1));

export class JsTrainer {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsTrainerFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jstrainer_free(ptr, 0);
    }
    /**
     * @param {JsModelConfig} model_config
     * @param {JsTrainingConfig} training_config
     */
    constructor(model_config, training_config) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(model_config, JsModelConfig);
            _assertClass(training_config, JsTrainingConfig);
            wasm.jstrainer_new(retptr, model_config.__wbg_ptr, training_config.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            JsTrainerFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {JsDataset} dataset
     * @returns {Promise<JsTrainingResult>}
     */
    train(dataset) {
        _assertClass(dataset, JsDataset);
        const ret = wasm.jstrainer_train(this.__wbg_ptr, dataset.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @param {JsTensor} input
     * @returns {JsTensor}
     */
    predict(input) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(input, JsTensor);
            wasm.jstrainer_predict(retptr, this.__wbg_ptr, input.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTensor.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @returns {Array<any>}
     */
    weight_matrices() {
        const ret = wasm.jstrainer_weight_matrices(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {Array<any>}
     */
    bias_vectors() {
        const ret = wasm.jstrainer_bias_vectors(this.__wbg_ptr);
        return takeObject(ret);
    }
}
if (Symbol.dispose) JsTrainer.prototype[Symbol.dispose] = JsTrainer.prototype.free;

const JsTrainingConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jstrainingconfig_free(ptr >>> 0, 1));

export class JsTrainingConfig {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(JsTrainingConfig.prototype);
        obj.__wbg_ptr = ptr;
        JsTrainingConfigFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsTrainingConfigFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jstrainingconfig_free(ptr, 0);
    }
    /**
     * @param {number} epochs
     * @param {number} batch_size
     * @param {number} validation_split
     * @param {JsOptimizerConfig} optimizer_config
     * @param {JsRegularizationConfig | null} [regularization_config]
     */
    constructor(epochs, batch_size, validation_split, optimizer_config, regularization_config) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(optimizer_config, JsOptimizerConfig);
            var ptr0 = optimizer_config.__destroy_into_raw();
            let ptr1 = 0;
            if (!isLikeNone(regularization_config)) {
                _assertClass(regularization_config, JsRegularizationConfig);
                ptr1 = regularization_config.__destroy_into_raw();
            }
            wasm.jstrainingconfig_new(retptr, epochs, batch_size, validation_split, ptr0, ptr1);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            JsTrainingConfigFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} epochs
     * @param {number} batch_size
     * @param {number} validation_split
     * @param {JsOptimizerConfig} optimizer_config
     * @param {JsRegularizationConfig | null | undefined} regularization_config
     * @param {boolean} enable_early_stopping
     * @param {number} early_stopping_patience
     * @param {number} early_stopping_min_delta
     * @returns {JsTrainingConfig}
     */
    static newWithEarlyStopping(epochs, batch_size, validation_split, optimizer_config, regularization_config, enable_early_stopping, early_stopping_patience, early_stopping_min_delta) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(optimizer_config, JsOptimizerConfig);
            var ptr0 = optimizer_config.__destroy_into_raw();
            let ptr1 = 0;
            if (!isLikeNone(regularization_config)) {
                _assertClass(regularization_config, JsRegularizationConfig);
                ptr1 = regularization_config.__destroy_into_raw();
            }
            wasm.jstrainingconfig_newWithEarlyStopping(retptr, epochs, batch_size, validation_split, ptr0, ptr1, enable_early_stopping, early_stopping_patience, early_stopping_min_delta);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return JsTrainingConfig.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) JsTrainingConfig.prototype[Symbol.dispose] = JsTrainingConfig.prototype.free;

const JsTrainingResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jstrainingresult_free(ptr >>> 0, 1));

export class JsTrainingResult {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(JsTrainingResult.prototype);
        obj.__wbg_ptr = ptr;
        JsTrainingResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JsTrainingResultFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jstrainingresult_free(ptr, 0);
    }
    /**
     * @returns {Array<any>}
     */
    get loss_history() {
        const ret = wasm.jstrainingresult_loss_history(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {Array<any>}
     */
    get accuracy_history() {
        const ret = wasm.jstrainingresult_accuracy_history(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {Array<any>}
     */
    get validation_loss_history() {
        const ret = wasm.jstrainingresult_validation_loss_history(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {Array<any>}
     */
    get validation_accuracy_history() {
        const ret = wasm.jstrainingresult_validation_accuracy_history(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * @returns {JsMetrics}
     */
    get final_metrics() {
        const ret = wasm.jstrainingresult_final_metrics(this.__wbg_ptr);
        return JsMetrics.__wrap(ret);
    }
}
if (Symbol.dispose) JsTrainingResult.prototype[Symbol.dispose] = JsTrainingResult.prototype.free;

const UtilsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_utils_free(ptr >>> 0, 1));
/**
 * Utility functions
 */
export class Utils {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        UtilsFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_utils_free(ptr, 0);
    }
    /**
     * Create a simple 2-layer neural network for binary classification
     * @param {number} input_size
     * @param {number} hidden_size
     * @returns {JsModel}
     */
    static create_binary_classifier(input_size, hidden_size) {
        const ret = wasm.utils_create_binary_classifier(input_size, hidden_size);
        return JsModel.__wrap(ret);
    }
    /**
     * Create a multi-class classifier with softmax output
     * @param {number} input_size
     * @param {number} hidden_size
     * @param {number} num_classes
     * @returns {JsModel}
     */
    static create_multiclass_classifier(input_size, hidden_size, num_classes) {
        const ret = wasm.utils_create_multiclass_classifier(input_size, hidden_size, num_classes);
        return JsModel.__wrap(ret);
    }
    /**
     * Log a message to the browser console
     * @param {string} message
     */
    static log(message) {
        const ptr0 = passStringToWasm0(message, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        wasm.utils_log(ptr0, len0);
    }
    /**
     * Calculate binary cross-entropy loss
     * @param {JsTensor} predictions
     * @param {JsTensor} targets
     * @returns {number}
     */
    static binary_cross_entropy(predictions, targets) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(predictions, JsTensor);
            _assertClass(targets, JsTensor);
            wasm.utils_binary_cross_entropy(retptr, predictions.__wbg_ptr, targets.__wbg_ptr);
            var r0 = getDataViewMemory0().getFloat64(retptr + 8 * 0, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            if (r3) {
                throw takeObject(r2);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Calculate accuracy for binary classification
     * @param {JsTensor} predictions
     * @param {JsTensor} targets
     * @returns {number}
     */
    static binary_accuracy(predictions, targets) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            _assertClass(predictions, JsTensor);
            _assertClass(targets, JsTensor);
            wasm.utils_binary_accuracy(retptr, predictions.__wbg_ptr, targets.__wbg_ptr);
            var r0 = getDataViewMemory0().getFloat64(retptr + 8 * 0, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            if (r3) {
                throw takeObject(r2);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) Utils.prototype[Symbol.dispose] = Utils.prototype.free;

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
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
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg___wbindgen_is_function_ee8a6c5833c90377 = function(arg0) {
        const ret = typeof(getObject(arg0)) === 'function';
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_object_c818261d21f283a4 = function(arg0) {
        const val = getObject(arg0);
        const ret = typeof(val) === 'object' && val !== null;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_string_fbb76cb2940daafd = function(arg0) {
        const ret = typeof(getObject(arg0)) === 'string';
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_undefined_2d472862bd29a478 = function(arg0) {
        const ret = getObject(arg0) === undefined;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_number_get_a20bf9b85341449d = function(arg0, arg1) {
        const obj = getObject(arg1);
        const ret = typeof(obj) === 'number' ? obj : undefined;
        getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    };
    imports.wbg.__wbg___wbindgen_throw_b855445ff6a94295 = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg__wbg_cb_unref_2454a539ea5790d9 = function(arg0) {
        getObject(arg0)._wbg_cb_unref();
    };
    imports.wbg.__wbg_call_525440f72fbfc0ea = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = getObject(arg0).call(getObject(arg1), getObject(arg2));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_call_e762c39fa8ea36bf = function() { return handleError(function (arg0, arg1) {
        const ret = getObject(arg0).call(getObject(arg1));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_crypto_574e78ad8b13b65f = function(arg0) {
        const ret = getObject(arg0).crypto;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_error_7534b8e9a36f1ab4 = function(arg0, arg1) {
        let deferred0_0;
        let deferred0_1;
        try {
            deferred0_0 = arg0;
            deferred0_1 = arg1;
            console.error(getStringFromWasm0(arg0, arg1));
        } finally {
            wasm.__wbindgen_export2(deferred0_0, deferred0_1, 1);
        }
    };
    imports.wbg.__wbg_getRandomValues_b8f5dbd5f3995a9e = function() { return handleError(function (arg0, arg1) {
        getObject(arg0).getRandomValues(getObject(arg1));
    }, arguments) };
    imports.wbg.__wbg_get_7bed016f185add81 = function(arg0, arg1) {
        const ret = getObject(arg0)[arg1 >>> 0];
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_get_efcb449f58ec27c2 = function() { return handleError(function (arg0, arg1) {
        const ret = Reflect.get(getObject(arg0), getObject(arg1));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_instanceof_Float64Array_4d61421d674c37cb = function(arg0) {
        let result;
        try {
            result = getObject(arg0) instanceof Float64Array;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_isArray_96e0af9891d0945d = function(arg0) {
        const ret = Array.isArray(getObject(arg0));
        return ret;
    };
    imports.wbg.__wbg_jstrainingresult_new = function(arg0) {
        const ret = JsTrainingResult.__wrap(arg0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_length_69bca3cb64fc8748 = function(arg0) {
        const ret = getObject(arg0).length;
        return ret;
    };
    imports.wbg.__wbg_length_cdd215e10d9dd507 = function(arg0) {
        const ret = getObject(arg0).length;
        return ret;
    };
    imports.wbg.__wbg_length_e70e9e6484b0952f = function(arg0) {
        const ret = getObject(arg0).length;
        return ret;
    };
    imports.wbg.__wbg_log_8cec76766b8c0e33 = function(arg0) {
        console.log(getObject(arg0));
    };
    imports.wbg.__wbg_log_9673b50e005015de = function(arg0, arg1) {
        console.log(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_msCrypto_a61aeb35a24c1329 = function(arg0) {
        const ret = getObject(arg0).msCrypto;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_3c3d849046688a66 = function(arg0, arg1) {
        try {
            var state0 = {a: arg0, b: arg1};
            var cb0 = (arg0, arg1) => {
                const a = state0.a;
                state0.a = 0;
                try {
                    return __wasm_bindgen_func_elem_1210(a, state0.b, arg0, arg1);
                } finally {
                    state0.a = a;
                }
            };
            const ret = new Promise(cb0);
            return addHeapObject(ret);
        } finally {
            state0.a = state0.b = 0;
        }
    };
    imports.wbg.__wbg_new_8a6f238a6ece86ea = function() {
        const ret = new Error();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_e17d9f43105b08be = function() {
        const ret = new Array();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_from_slice_fde3e31e670b38a6 = function(arg0, arg1) {
        const ret = new Float64Array(getArrayF64FromWasm0(arg0, arg1));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_no_args_ee98eee5275000a4 = function(arg0, arg1) {
        const ret = new Function(getStringFromWasm0(arg0, arg1));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_with_length_01aa0dc35aa13543 = function(arg0) {
        const ret = new Uint8Array(arg0 >>> 0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_node_905d3e251edff8a2 = function(arg0) {
        const ret = getObject(arg0).node;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_process_dc0fbacc7c1c06f7 = function(arg0) {
        const ret = getObject(arg0).process;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_prototypesetcall_2a6620b6922694b2 = function(arg0, arg1, arg2) {
        Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), getObject(arg2));
    };
    imports.wbg.__wbg_prototypesetcall_31bbb896072c2bfc = function(arg0, arg1, arg2) {
        Float64Array.prototype.set.call(getArrayF64FromWasm0(arg0, arg1), getObject(arg2));
    };
    imports.wbg.__wbg_push_df81a39d04db858c = function(arg0, arg1) {
        const ret = getObject(arg0).push(getObject(arg1));
        return ret;
    };
    imports.wbg.__wbg_queueMicrotask_34d692c25c47d05b = function(arg0) {
        const ret = getObject(arg0).queueMicrotask;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_queueMicrotask_9d76cacb20c84d58 = function(arg0) {
        queueMicrotask(getObject(arg0));
    };
    imports.wbg.__wbg_randomFillSync_ac0988aba3254290 = function() { return handleError(function (arg0, arg1) {
        getObject(arg0).randomFillSync(takeObject(arg1));
    }, arguments) };
    imports.wbg.__wbg_require_60cc747a6bc5215a = function() { return handleError(function () {
        const ret = module.require;
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_resolve_caf97c30b83f7053 = function(arg0) {
        const ret = Promise.resolve(getObject(arg0));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_stack_0ed75d68575b0f3c = function(arg0, arg1) {
        const ret = getObject(arg1).stack;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_89e1d9ac6a1b250e = function() {
        const ret = typeof global === 'undefined' ? null : global;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_THIS_8b530f326a9e48ac = function() {
        const ret = typeof globalThis === 'undefined' ? null : globalThis;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbg_static_accessor_SELF_6fdf4b64710cc91b = function() {
        const ret = typeof self === 'undefined' ? null : self;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbg_static_accessor_WINDOW_b45bfc5a37f6cfa2 = function() {
        const ret = typeof window === 'undefined' ? null : window;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbg_subarray_480600f3d6a9f26c = function(arg0, arg1, arg2) {
        const ret = getObject(arg0).subarray(arg1 >>> 0, arg2 >>> 0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_then_4f46f6544e6b4a28 = function(arg0, arg1) {
        const ret = getObject(arg0).then(getObject(arg1));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_versions_c01dfd4722a88165 = function(arg0) {
        const ret = getObject(arg0).versions;
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_cast_cb9088102bce6b30 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
        const ret = getArrayU8FromWasm0(arg0, arg1);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_cast_d6cd19b81560fd6e = function(arg0) {
        // Cast intrinsic for `F64 -> Externref`.
        const ret = arg0;
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_cast_f0471e20a57dc27a = function(arg0, arg1) {
        // Cast intrinsic for `Closure(Closure { dtor_idx: 127, function: Function { arguments: [Externref], shim_idx: 128, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
        const ret = makeMutClosure(arg0, arg1, wasm.__wasm_bindgen_func_elem_826, __wasm_bindgen_func_elem_830);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_object_clone_ref = function(arg0) {
        const ret = getObject(arg0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_object_drop_ref = function(arg0) {
        takeObject(arg0);
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
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


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('multilayer_perceptron_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
