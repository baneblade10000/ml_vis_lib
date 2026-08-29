/* tslint:disable */
/* eslint-disable */
export function main(): void;
export function generate_dataset_from_points(points: Array<any>): JsDataset;
export enum JsOptimizerType {
  GD = 0,
  SGD = 1,
  SGDMomentum = 2,
  RMSProp = 3,
  Adam = 4,
}
export enum JsRegularizationType {
  None = 0,
  L1 = 1,
  L2 = 2,
  ElasticNet = 3,
}
export enum JsTaskType {
  BinaryClassification = 0,
  MultiClassification = 1,
  Regression = 2,
}
/**
 * Data conversion utilities between JavaScript and Rust types
 */
export class DataConverter {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Convert JavaScript 2D array to JsTensor
   */
  static array_to_tensor(array: Array<any>): JsTensor;
  /**
   * Convert JsTensor to JavaScript 2D array
   */
  static tensor_to_array(tensor: JsTensor): Array<any>;
  /**
   * Convert flat JavaScript array to JsTensor with specified shape
   */
  static flat_array_to_tensor(flat_array: Float64Array, rows: number, cols: number): JsTensor;
  /**
   * Convert JsTensor to flat JavaScript array
   */
  static tensor_to_flat_array(tensor: JsTensor): Float64Array;
  /**
   * Create tensor from CSV-like string data
   */
  static csv_to_tensor(csv_string: string, has_header: boolean, delimiter: string): JsTensor;
  /**
   * Convert tensor to CSV-like string
   */
  static tensor_to_csv(tensor: JsTensor, delimiter: string): string;
  /**
   * Normalize tensor values to [0, 1] range
   */
  static normalize_min_max(tensor: JsTensor): JsTensor;
  /**
   * Standardize tensor values (z-score normalization)
   */
  static standardize(tensor: JsTensor): JsTensor;
}
export class JsDataPoint {
  free(): void;
  [Symbol.dispose](): void;
  constructor(x: number, y: number, label: number);
  readonly x: number;
  readonly y: number;
  readonly label: number;
}
/**
 * Dataset wrapper for JavaScript
 */
export class JsDataset {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new dataset from features and labels (defaulting to binary classification)
   */
  constructor(features: Array<any>, labels: Float64Array);
  /**
   * Get the number of samples
   */
  len(): number;
  /**
   * Get the number of features per sample
   */
  feature_count(): number;
  /**
   * Get features as a tensor
   */
  features_tensor(): JsTensor;
  /**
   * Get labels as a tensor
   */
  labels_tensor(): JsTensor;
  readonly task_type: JsTaskType;
}
export class JsMetrics {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  readonly accuracy: number;
  readonly loss: number;
  readonly precision: number | undefined;
  readonly recall: number | undefined;
  readonly f1_score: number | undefined;
  readonly mse: number | undefined;
}
/**
 * JavaScript-compatible wrapper for Neural Network Model
 */
export class JsModel {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new empty model
   */
  constructor();
  /**
   * Add a dense layer with ReLU activation
   */
  add_dense_relu(input_size: number, output_size: number): void;
  /**
   * Add a dense layer with sigmoid activation
   */
  add_dense_sigmoid(input_size: number, output_size: number): void;
  /**
   * Add a dense layer with softmax activation (typically for output layer)
   */
  add_dense_softmax(input_size: number, output_size: number): void;
  /**
   * Forward pass through the model
   */
  forward(input: JsTensor): JsTensor;
  /**
   * Get model summary as a string
   */
  summary(): string;
  /**
   * Get total number of parameters
   */
  param_count(): number;
}
export class JsModelConfig {
  free(): void;
  [Symbol.dispose](): void;
  constructor(layers: Array<any>, activation_fn: string, task_type: JsTaskType);
  readonly task_type: JsTaskType;
}
export class JsOptimizerConfig {
  free(): void;
  [Symbol.dispose](): void;
  constructor(optimizer_type: JsOptimizerType, learning_rate: number);
  readonly optimizer_type: JsOptimizerType;
  readonly learning_rate: number;
}
export class JsRegularizationConfig {
  free(): void;
  [Symbol.dispose](): void;
  constructor(reg_type: JsRegularizationType, l1_lambda: number, l2_lambda: number);
}
/**
 * JavaScript-compatible wrapper for Tensor
 */
export class JsTensor {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new tensor from a JavaScript Float64Array
   *
   * # Arguments
   * * `data` - Flattened tensor data as Float64Array
   * * `rows` - Number of rows
   * * `cols` - Number of columns
   */
  constructor(data: Float64Array, rows: number, cols: number);
  /**
   * Create a tensor filled with zeros
   */
  static zeros(rows: number, cols: number): JsTensor;
  /**
   * Create a tensor filled with ones
   */
  static ones(rows: number, cols: number): JsTensor;
  /**
   * Create a tensor with random values between -1 and 1
   */
  static random(rows: number, cols: number): JsTensor;
  /**
   * Get the shape of the tensor as [rows, cols]
   */
  shape(): Array<any>;
  /**
   * Get the tensor data as a flattened Float64Array
   */
  data(): Float64Array;
  /**
   * Set whether this tensor requires gradients
   */
  set_requires_grad(requires_grad: boolean): void;
  /**
   * Check if this tensor requires gradients
   */
  requires_grad(): boolean;
  /**
   * Get the gradient as a JsTensor (if available)
   */
  gradient(): JsTensor | undefined;
  /**
   * Zero out the gradients
   */
  zero_grad(): void;
  /**
   * Perform matrix multiplication
   */
  matmul(other: JsTensor): JsTensor;
  /**
   * Add two tensors
   */
  add(other: JsTensor): JsTensor;
  /**
   * Subtract two tensors
   */
  sub(other: JsTensor): JsTensor;
  /**
   * Element-wise multiplication
   */
  mul(other: JsTensor): JsTensor;
  /**
   * Scalar multiplication
   */
  mul_scalar(scalar: number): JsTensor;
  /**
   * Apply sigmoid activation
   */
  sigmoid(): JsTensor;
  /**
   * Apply ReLU activation
   */
  relu(): JsTensor;
  /**
   * Apply tanh activation
   */
  tanh(): JsTensor;
  /**
   * Apply softmax activation
   */
  softmax(): JsTensor;
  /**
   * Compute mean of all elements
   */
  mean(): JsTensor;
  /**
   * Perform backward pass (compute gradients)
   */
  backward(): void;
  /**
   * Clone the tensor
   */
  clone(): JsTensor;
  /**
   * Get a string representation of the tensor
   */
  to_string(): string;
  /**
   * Log the tensor to browser console (for debugging)
   */
  log(): void;
}
export class JsTrainer {
  free(): void;
  [Symbol.dispose](): void;
  constructor(model_config: JsModelConfig, training_config: JsTrainingConfig);
  train(dataset: JsDataset): Promise<JsTrainingResult>;
  predict(input: JsTensor): JsTensor;
  weight_matrices(): Array<any>;
  bias_vectors(): Array<any>;
}
export class JsTrainingConfig {
  free(): void;
  [Symbol.dispose](): void;
  constructor(epochs: number, batch_size: number, validation_split: number, optimizer_config: JsOptimizerConfig, regularization_config?: JsRegularizationConfig | null);
  static newWithEarlyStopping(epochs: number, batch_size: number, validation_split: number, optimizer_config: JsOptimizerConfig, regularization_config: JsRegularizationConfig | null | undefined, enable_early_stopping: boolean, early_stopping_patience: number, early_stopping_min_delta: number): JsTrainingConfig;
}
export class JsTrainingResult {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  readonly loss_history: Array<any>;
  readonly accuracy_history: Array<any>;
  readonly validation_loss_history: Array<any>;
  readonly validation_accuracy_history: Array<any>;
  readonly final_metrics: JsMetrics;
}
/**
 * Utility functions
 */
export class Utils {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a simple 2-layer neural network for binary classification
   */
  static create_binary_classifier(input_size: number, hidden_size: number): JsModel;
  /**
   * Create a multi-class classifier with softmax output
   */
  static create_multiclass_classifier(input_size: number, hidden_size: number, num_classes: number): JsModel;
  /**
   * Log a message to the browser console
   */
  static log(message: string): void;
  /**
   * Calculate binary cross-entropy loss
   */
  static binary_cross_entropy(predictions: JsTensor, targets: JsTensor): number;
  /**
   * Calculate accuracy for binary classification
   */
  static binary_accuracy(predictions: JsTensor, targets: JsTensor): number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_jsoptimizerconfig_free: (a: number, b: number) => void;
  readonly jsoptimizerconfig_new: (a: number, b: number, c: number) => void;
  readonly jsoptimizerconfig_optimizer_type: (a: number) => number;
  readonly jsoptimizerconfig_learning_rate: (a: number) => number;
  readonly jsregularizationconfig_new: (a: number, b: number, c: number, d: number) => void;
  readonly __wbg_jsmodelconfig_free: (a: number, b: number) => void;
  readonly jsmodelconfig_new: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly jsmodelconfig_task_type: (a: number) => number;
  readonly __wbg_jstrainingconfig_free: (a: number, b: number) => void;
  readonly jstrainingconfig_new: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly jstrainingconfig_newWithEarlyStopping: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
  readonly main: () => void;
  readonly __wbg_jstensor_free: (a: number, b: number) => void;
  readonly jstensor_zeros: (a: number, b: number) => number;
  readonly jstensor_ones: (a: number, b: number) => number;
  readonly jstensor_random: (a: number, b: number) => number;
  readonly jstensor_shape: (a: number) => number;
  readonly jstensor_set_requires_grad: (a: number, b: number) => void;
  readonly jstensor_requires_grad: (a: number) => number;
  readonly jstensor_gradient: (a: number) => number;
  readonly jstensor_zero_grad: (a: number) => void;
  readonly jstensor_matmul: (a: number, b: number, c: number) => void;
  readonly jstensor_add: (a: number, b: number, c: number) => void;
  readonly jstensor_sub: (a: number, b: number, c: number) => void;
  readonly jstensor_mul: (a: number, b: number, c: number) => void;
  readonly jstensor_mul_scalar: (a: number, b: number, c: number) => void;
  readonly jstensor_sigmoid: (a: number, b: number) => void;
  readonly jstensor_relu: (a: number, b: number) => void;
  readonly jstensor_tanh: (a: number, b: number) => void;
  readonly jstensor_softmax: (a: number, b: number) => void;
  readonly jstensor_mean: (a: number, b: number) => void;
  readonly jstensor_backward: (a: number, b: number) => void;
  readonly jstensor_clone: (a: number) => number;
  readonly jstensor_to_string: (a: number, b: number) => void;
  readonly jstensor_log: (a: number) => void;
  readonly __wbg_jsmodel_free: (a: number, b: number) => void;
  readonly jsmodel_new: () => number;
  readonly jsmodel_add_dense_relu: (a: number, b: number, c: number, d: number) => void;
  readonly jsmodel_add_dense_sigmoid: (a: number, b: number, c: number, d: number) => void;
  readonly jsmodel_add_dense_softmax: (a: number, b: number, c: number, d: number) => void;
  readonly jsmodel_forward: (a: number, b: number, c: number) => void;
  readonly jsmodel_summary: (a: number, b: number) => void;
  readonly jsmodel_param_count: (a: number) => number;
  readonly __wbg_jstrainer_free: (a: number, b: number) => void;
  readonly jstrainer_new: (a: number, b: number, c: number) => void;
  readonly jstrainer_train: (a: number, b: number) => number;
  readonly jstrainer_predict: (a: number, b: number, c: number) => void;
  readonly jstrainer_weight_matrices: (a: number) => number;
  readonly jstrainer_bias_vectors: (a: number) => number;
  readonly __wbg_jstrainingresult_free: (a: number, b: number) => void;
  readonly jstrainingresult_loss_history: (a: number) => number;
  readonly jstrainingresult_accuracy_history: (a: number) => number;
  readonly jstrainingresult_validation_loss_history: (a: number) => number;
  readonly jstrainingresult_validation_accuracy_history: (a: number) => number;
  readonly jstrainingresult_final_metrics: (a: number) => number;
  readonly __wbg_jsmetrics_free: (a: number, b: number) => void;
  readonly jsmetrics_accuracy: (a: number) => number;
  readonly jsmetrics_loss: (a: number) => number;
  readonly jsmetrics_precision: (a: number, b: number) => void;
  readonly jsmetrics_recall: (a: number, b: number) => void;
  readonly jsmetrics_f1_score: (a: number, b: number) => void;
  readonly jsmetrics_mse: (a: number, b: number) => void;
  readonly __wbg_jsdatapoint_free: (a: number, b: number) => void;
  readonly jsdatapoint_new: (a: number, b: number, c: number) => number;
  readonly jsdatapoint_x: (a: number) => number;
  readonly jsdatapoint_y: (a: number) => number;
  readonly jsdatapoint_label: (a: number) => number;
  readonly __wbg_jsdataset_free: (a: number, b: number) => void;
  readonly jsdataset_new: (a: number, b: number, c: number) => void;
  readonly jsdataset_task_type: (a: number) => number;
  readonly jsdataset_len: (a: number) => number;
  readonly jsdataset_feature_count: (a: number) => number;
  readonly jsdataset_features_tensor: (a: number, b: number) => void;
  readonly jsdataset_labels_tensor: (a: number, b: number) => void;
  readonly generate_dataset_from_points: (a: number, b: number) => void;
  readonly __wbg_dataconverter_free: (a: number, b: number) => void;
  readonly dataconverter_array_to_tensor: (a: number, b: number) => void;
  readonly dataconverter_tensor_to_array: (a: number) => number;
  readonly dataconverter_flat_array_to_tensor: (a: number, b: number, c: number, d: number) => void;
  readonly dataconverter_tensor_to_flat_array: (a: number) => number;
  readonly dataconverter_csv_to_tensor: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly dataconverter_tensor_to_csv: (a: number, b: number, c: number, d: number) => void;
  readonly dataconverter_normalize_min_max: (a: number, b: number) => void;
  readonly dataconverter_standardize: (a: number, b: number) => void;
  readonly utils_create_binary_classifier: (a: number, b: number) => number;
  readonly utils_create_multiclass_classifier: (a: number, b: number, c: number) => number;
  readonly utils_log: (a: number, b: number) => void;
  readonly utils_binary_cross_entropy: (a: number, b: number, c: number) => void;
  readonly utils_binary_accuracy: (a: number, b: number, c: number) => void;
  readonly jstensor_new: (a: number, b: number, c: number, d: number) => void;
  readonly jstensor_data: (a: number) => number;
  readonly __wbg_utils_free: (a: number, b: number) => void;
  readonly __wbg_jsregularizationconfig_free: (a: number, b: number) => void;
  readonly __wasm_bindgen_func_elem_830: (a: number, b: number, c: number) => void;
  readonly __wasm_bindgen_func_elem_826: (a: number, b: number) => void;
  readonly __wasm_bindgen_func_elem_1210: (a: number, b: number, c: number, d: number) => void;
  readonly __wbindgen_export: (a: number) => void;
  readonly __wbindgen_export2: (a: number, b: number, c: number) => void;
  readonly __wbindgen_export3: (a: number, b: number) => number;
  readonly __wbindgen_export4: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
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
