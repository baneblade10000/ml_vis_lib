"use client"

import { useCallback, useRef, useState } from "react"
import type { DataPoint } from "@/components/dataset-generator"

type OptimizerType = "gd" | "sgd" | "sgd_momentum" | "rmsprop" | "adam"
type RegularizationType = "none" | "l1" | "l2" | "elastic_net"
type TaskType = "binary" | "multi" | "regression"

interface RegularizationConfigState {
  type: RegularizationType
  l1: number
  l2: number
}

export interface StartTrainingOptions {
  layers: number[]
  activationFn: string
  optimizer: OptimizerType
  learningRate: number
  regularization: RegularizationConfigState
  dataset: DataPoint[]
  epochs: number
  batchSize: number
  validationSplit: number
  taskType: TaskType
  enableEarlyStopping?: boolean
  earlyStoppingPatience?: number
  earlyStoppingMinDelta?: number
}

export interface TrainingHistoryState {
  loss: number[]
  accuracy: number[]
  valLoss: (number | null)[]
  valAccuracy: (number | null)[]
}

export interface TrainingMetricsState {
  loss: number
  accuracy?: number
  precision?: number
  recall?: number
  f1Score?: number
  mse?: number
}

const INITIAL_LOGS = ["> Neural network initialized", "> Ready to train"]

const toNumberArray = (value: any): number[] => Array.from(value as unknown as Iterable<number>).map((item) => Number(item))

const toNullableNumberArray = (value: any): (number | null)[] =>
  toNumberArray(value).map((item) => (Number.isFinite(item) ? item : null))

const toNestedNumberArray = (value: any): number[][][] =>
  Array.from(value as unknown as Iterable<any>).map((layer) =>
    Array.from(layer as Iterable<any>).map((row) => Array.from(row as Iterable<any>).map((entry) => Number(entry))),
  )

type WasmModule = typeof import("../../pkg/multilayer_perceptron")

export function useMLTraining(wasmModule: WasmModule | null) {
  const trainerRef = useRef<InstanceType<WasmModule["JsTrainer"]> | null>(null)

  const [history, setHistory] = useState<TrainingHistoryState | null>(null)
  const [metrics, setMetrics] = useState<TrainingMetricsState | null>(null)
  const [weights, setWeights] = useState<number[][][] | null>(null)
  const [isTraining, setIsTraining] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>(INITIAL_LOGS)
  const [hasModel, setHasModel] = useState(false)

  const appendLogs = useCallback((entries: string[]) => {
    setLogs((prev) => {
      const merged = [...prev, ...entries]
      return merged.slice(-60)
    })
  }, [])

  const startTraining = useCallback(
    async (options: StartTrainingOptions) => {
      if (!wasmModule) {
        setError("WASM module is not ready")
        return
      }

      if (!options.dataset.length) {
        setError("Generate a dataset before training")
        return
      }

      setIsTraining(true)
      setError(null)
      setHasModel(false)
      appendLogs([
        `> Starting training for ${options.epochs} epochs`,
      ])

      try {
        const sanitizedLayers = options.layers.length >= 2 ? [...options.layers] : [options.layers[0] || 2, 1]
        const featureCount = 2 // generated dataset currently uses 2D points (x, y)
        sanitizedLayers[0] = featureCount

        const featureMatrix = options.dataset.map((point) => new Float64Array([point.x, point.y]))
        const labelVector = new Float64Array(options.dataset.map((point) => point.label))
        const dataset = new wasmModule.JsDataset(featureMatrix as unknown as any[], labelVector)

        const optimizerType = (() => {
          switch (options.optimizer) {
            case "gd":
              return wasmModule.JsOptimizerType.GD
            case "sgd":
              return wasmModule.JsOptimizerType.SGD
            case "sgd_momentum":
              return wasmModule.JsOptimizerType.SGDMomentum
            case "rmsprop":
              return wasmModule.JsOptimizerType.RMSProp
            case "adam":
            default:
              return wasmModule.JsOptimizerType.Adam
          }
        })()

        const taskType = (() => {
          switch (options.taskType) {
            case "multi":
              return wasmModule.JsTaskType.MultiClassification
            case "regression":
              return wasmModule.JsTaskType.Regression
            case "binary":
            default:
              return wasmModule.JsTaskType.BinaryClassification
          }
        })()

        const regularizationType = (() => {
          switch (options.regularization.type) {
            case "l1":
              return wasmModule.JsRegularizationType.L1
            case "l2":
              return wasmModule.JsRegularizationType.L2
            case "elastic_net":
              return wasmModule.JsRegularizationType.ElasticNet
            case "none":
            default:
              return wasmModule.JsRegularizationType.None
          }
        })()

        const optimizerConfig = new wasmModule.JsOptimizerConfig(optimizerType, options.learningRate)
        const regularizationConfig =
          options.regularization.type === "none"
            ? undefined
            : new wasmModule.JsRegularizationConfig(
                regularizationType,
                options.regularization.l1,
                options.regularization.l2,
              )

        const trainingConfig = options.enableEarlyStopping
          ? (wasmModule as any).JsTrainingConfig.newWithEarlyStopping(
              Math.max(1, Math.floor(options.epochs)),
              Math.max(1, Math.floor(options.batchSize)),
              Math.min(Math.max(options.validationSplit, 0.1), 0.9), // Early stopping requires validation data
              optimizerConfig,
              regularizationConfig,
              options.enableEarlyStopping,
              options.earlyStoppingPatience || 5,
              options.earlyStoppingMinDelta || 0.001,
            )
          : new wasmModule.JsTrainingConfig(
              Math.max(1, Math.floor(options.epochs)),
              Math.max(1, Math.floor(options.batchSize)),
              Math.min(Math.max(options.validationSplit, 0), 0.9),
              optimizerConfig,
              regularizationConfig,
            )

        const modelConfig = new wasmModule.JsModelConfig(
          sanitizedLayers,
          options.activationFn.toLowerCase(),
          taskType,
        )

        const trainer = new wasmModule.JsTrainer(modelConfig, trainingConfig)
        trainerRef.current = trainer

        const result = await trainer.train(dataset)

        const lossHistory = toNumberArray((result as any).loss_history)
        const accuracyHistory = toNumberArray((result as any).accuracy_history)
        const valLossHistory = toNullableNumberArray((result as any).validation_loss_history || [])
        const valAccuracyHistory = toNullableNumberArray((result as any).validation_accuracy_history || [])

        setHistory({
          loss: lossHistory,
          accuracy: accuracyHistory,
          valLoss: valLossHistory,
          valAccuracy: valAccuracyHistory,
        })

        const finalMetrics = (result as any).final_metrics
        setMetrics({
          loss: finalMetrics.loss || 0,
          accuracy: finalMetrics.accuracy,
          precision: finalMetrics.precision,
          recall: finalMetrics.recall,
          f1Score: finalMetrics.f1_score,
          mse: finalMetrics.mse,
        })

        try {
          const weightSnapshot = (trainer as any).weight_matrices ? (trainer as any).weight_matrices() : null
          setWeights(weightSnapshot ? toNestedNumberArray(weightSnapshot) : null)
        } catch (error) {
          console.log("Could not retrieve weights:", error)
          setWeights(null)
        }

        const epochLogs = lossHistory.map((loss, index) => {
          const acc = accuracyHistory[index]
          const lossText = Number.isFinite(loss) ? loss.toFixed(4) : "—"
          const accText = Number.isFinite(acc) ? `${(acc * 100).toFixed(2)}%` : "—"
          return `> Epoch ${index + 1}: loss=${lossText} accuracy=${accText}`
        })

        appendLogs([...epochLogs, "> Training complete"])
        setHasModel(true)
      } catch (err) {
        console.error("Training failed", err)
        const message = err instanceof Error ? err.message : "Training failed"
        setError(message)
        appendLogs([`> Error: ${message}`])
        trainerRef.current = null
        setHasModel(false)
        throw err
      } finally {
        setIsTraining(false)
      }
    },
    [appendLogs, wasmModule],
  )

  const resetTraining = useCallback(() => {
    trainerRef.current = null
    setHistory(null)
    setMetrics(null)
    setWeights(null)
    setError(null)
    setLogs(INITIAL_LOGS)
    setHasModel(false)
  }, [])

  const predict = useCallback(
    (points: { x: number; y: number }[]): number[] => {
      if (!wasmModule || !trainerRef.current || !hasModel || points.length === 0) {
        return []
      }

      const rows = points.length
      const flat = new Float64Array(rows * 2)

      points.forEach((point, index) => {
        flat[index * 2] = point.x
        flat[index * 2 + 1] = point.y
      })

      const tensor = new wasmModule.JsTensor(flat, rows, 2)
      const predictor = (trainerRef.current as any).predict
      if (typeof predictor !== "function") {
        return []
      }
      const output = predictor.call(trainerRef.current, tensor)
      const result = output.data()
      const raw = Array.from(result)
      if (rows === 0) {
        return []
      }

      const outputsPerSample = raw.length / rows
      if (!Number.isFinite(outputsPerSample) || outputsPerSample <= 1) {
        return raw.map((value) => (Number.isFinite(Number(value)) ? Math.max(0, Math.min(1, Number(value))) : 0))
      }

      const aggregated: number[] = []
      for (let row = 0; row < rows; row++) {
        let maxValue = -Infinity
        let maxIndex = 0

        for (let col = 0; col < outputsPerSample; col++) {
          const value = Number(raw[row * outputsPerSample + col])
          if (value > maxValue) {
            maxValue = value
            maxIndex = col
          }
        }

        aggregated.push(maxIndex === outputsPerSample - 1 ? 1 : 0)
      }

      return aggregated
    },
    [hasModel, wasmModule],
  )

  return {
    startTraining,
    resetTraining,
    isTraining,
    history,
    metrics,
    weights,
    error,
    logs,
    predict: hasModel ? predict : undefined,
    hasModel,
  }
}
