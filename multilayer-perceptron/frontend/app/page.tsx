"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { NetworkVisualizer } from "@/components/network-visualizer"
import { LayerControlPanel } from "@/components/layer-control-panel"
import { TrainingDashboard } from "@/components/training-dashboard"
import { OptimizerPanel } from "@/components/optimizer-panel"
import { ConsoleOutput } from "@/components/console-output"
import { NeuralBackground } from "@/components/neural-background"
import { DatasetGenerator, type DataPoint } from "@/components/dataset-generator"
import { useWASM } from "@/hooks/useWASM"
import { useMLTraining } from "@/hooks/useMLTraining"
import { Brain, Github, BookOpen } from "lucide-react"

export default function MLPDemo() {
  const [layers, setLayers] = useState([2, 8, 8, 1])
  const [activationFn, setActivationFn] = useState("relu")
  const [optimizer, setOptimizer] = useState<"gd" | "sgd" | "sgd_momentum" | "rmsprop" | "adam">("adam")
  const [learningRate, setLearningRate] = useState(0.01)
  const [regularization, setRegularization] = useState<"none" | "l1" | "l2" | "elastic_net">("none")
  const [l1Lambda, setL1Lambda] = useState(0.01)
  const [l2Lambda, setL2Lambda] = useState(0.01)
  const [dataset, setDataset] = useState<DataPoint[]>([])

  const { wasmModule, wasmError } = useWASM()
  const {
    startTraining,
    resetTraining,
    history,
    metrics,
    weights,
    isTraining,
    error: trainingError,
    logs,
    predict,
    hasModel,
  } = useMLTraining(wasmModule)
  const [decisionBoundary, setDecisionBoundary] = useState<{ cells: { x: number; y: number; value: number }[]; step: number } | null>(null)

  const handleDatasetGenerated = useCallback((points: DataPoint[]) => {
    setDataset(points)
    setDecisionBoundary(null)
    setLayers((prev) => {
      if (prev[0] === 2) {
        return prev
      }
      const updated = [...prev]
      updated[0] = 2
      return updated
    })
  }, [])

  const handleStartTraining = useCallback(
    ({ 
      epochs, 
      batchSize, 
      validationSplit, 
      enableEarlyStopping, 
      earlyStoppingPatience, 
      earlyStoppingMinDelta 
    }: { 
      epochs: number; 
      batchSize: number; 
      validationSplit: number;
      enableEarlyStopping: boolean;
      earlyStoppingPatience: number;
      earlyStoppingMinDelta: number;
    }) => {
      if (!dataset.length) {
        return
      }

      void startTraining({
        layers,
        activationFn,
        optimizer,
        learningRate,
        regularization: { type: regularization, l1: l1Lambda, l2: l2Lambda },
        dataset,
        epochs,
        batchSize,
        validationSplit,
        taskType: "binary",
        enableEarlyStopping,
        earlyStoppingPatience,
        earlyStoppingMinDelta,
      })
    },
    [activationFn, dataset, l1Lambda, l2Lambda, layers, learningRate, optimizer, regularization, startTraining],
  )

  const handleReset = useCallback(() => {
    resetTraining()
    setDecisionBoundary(null)
  }, [resetTraining])

  useEffect(() => {
    const predictor = predict
    if (!predictor || !hasModel || !history || history.loss.length === 0 || isTraining) {
      if (!predictor || !hasModel) {
        setDecisionBoundary(null)
      }
      return
    }

    const range = 6
    const resolution = 80
    const step = (range * 2) / (resolution - 1)
    const samples: { x: number; y: number }[] = []

    for (let ix = 0; ix < resolution; ix++) {
      for (let iy = 0; iy < resolution; iy++) {
        const x = -range + step * ix
        const y = -range + step * iy
        samples.push({ x, y })
      }
    }

    const values = predictor(samples)
    const cells = samples.map((point, index) => ({
      x: point.x,
      y: point.y,
      value: Number.isFinite(values[index]) ? values[index] : 0,
    }))

    setDecisionBoundary({ cells, step })
  }, [predict, hasModel, history, isTraining])

  const canTrain = useMemo(() => Boolean(wasmModule && dataset.length > 0), [dataset.length, wasmModule])
  const combinedError = trainingError ?? wasmError ?? null

  return (
    <div className="relative min-h-screen bg-background neural-grid overflow-hidden">
      <NeuralBackground />

      <div className="relative z-10">
        {/* Hero Section */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative border-b border-border/50 backdrop-blur-sm"
        >
          <div className="container mx-auto px-4 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <Brain className="w-8 h-8 text-primary animate-neural-pulse" />
                  <div className="absolute inset-0 blur-xl bg-primary/30 animate-pulse-glow" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-foreground">Neural Network Playground</h1>
                  <div className="flex items-center gap-2">
                    <p className="text-sm text-muted-foreground">Powered by</p>
                    <div className="flex items-center gap-1">
                      <span className="px-2 py-0.5 bg-orange-500/20 text-orange-400 text-xs font-mono rounded border border-orange-500/30">
                        Rust
                      </span>
                      <span className="text-muted-foreground text-xs">+</span>
                      <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 text-xs font-mono rounded border border-blue-500/30">
                        WebAssembly
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <a 
                  href="https://github.com/nktr-cp/multilayer-perceptron" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Github className="w-5 h-5" />
                </a>
              </div>
            </div>
          </div>
        </motion.header>

        <div className="container mx-auto px-4 py-8">
          <div className="space-y-6 max-w-7xl mx-auto">
            {/* Data Generation */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <span className="text-primary">01</span> Data Generation
              </h2>
              <div className="space-y-6">
                <DatasetGenerator onDatasetGenerated={handleDatasetGenerated} decisionBoundary={decisionBoundary ?? undefined} />
              </div>
            </motion.div>

            {/* Layer Controller / Optimizer */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <span className="text-primary">02</span> Layer Controller / Optimizer
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <LayerControlPanel
                  layers={layers}
                  onLayersChange={setLayers}
                  activationFn={activationFn}
                  onActivationChange={setActivationFn}
                />
                <OptimizerPanel
                  optimizer={optimizer}
                  onOptimizerChange={setOptimizer}
                  learningRate={learningRate}
                  onLearningRateChange={setLearningRate}
                  regularization={regularization}
                  onRegularizationChange={setRegularization}
                  l1Lambda={l1Lambda}
                  onL1LambdaChange={setL1Lambda}
                  l2Lambda={l2Lambda}
                  onL2LambdaChange={setL2Lambda}
                />
              </div>
            </motion.div>

            {/* Network Architecture */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <span className="text-primary">03</span> Network Architecture
              </h2>
              <NetworkVisualizer layers={layers} isTraining={isTraining} activationFn={activationFn} weights={weights} />
            </motion.div>

            {/* Training Metrics */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <span className="text-primary">04</span> Training Metrics
              </h2>
              <TrainingDashboard
                isTraining={isTraining}
                canTrain={canTrain}
                history={history}
                metrics={metrics}
                onStartTraining={handleStartTraining}
                onReset={handleReset}
                error={combinedError}
              />
            </motion.div>

            {/* Console */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
              <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <span className="text-primary">05</span> Console
              </h2>
              <ConsoleOutput isTraining={isTraining} logs={logs} />
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  )
}
