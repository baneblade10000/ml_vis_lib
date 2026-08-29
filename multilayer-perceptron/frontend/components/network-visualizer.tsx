"use client"

import { useEffect, useRef, useState } from "react"
import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"

interface NetworkVisualizerProps {
  layers: number[]
  isTraining: boolean
  activationFn: string
  weights?: number[][][] | null
}

export function NetworkVisualizer({ layers, isTraining, activationFn, weights }: NetworkVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [visualWeights, setVisualWeights] = useState<number[][][]>([])
  const weightsRef = useRef<number[][][]>(visualWeights)

  useEffect(() => {
    weightsRef.current = visualWeights
  }, [visualWeights])

  useEffect(() => {
    // Initialize random weights
    const newWeights: number[][][] = []
    for (let i = 0; i < layers.length - 1; i++) {
      const layerWeights: number[][] = []
      for (let j = 0; j < layers[i]; j++) {
        const neuronWeights: number[] = []
        for (let k = 0; k < layers[i + 1]; k++) {
          neuronWeights.push(Math.random() * 2 - 1)
        }
        layerWeights.push(neuronWeights)
      }
      newWeights.push(layerWeights)
    }
    setVisualWeights(newWeights)
  }, [layers])

  useEffect(() => {
    if (!weights || weights.length === 0) return
    setVisualWeights(weights)
  }, [weights])

  useEffect(() => {
    if (!canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const draw = () => {
      const width = canvas.width
      const height = canvas.height

      ctx.clearRect(0, 0, width, height)

      const layerSpacing = width / (layers.length + 1)
      const maxNeurons = Math.max(...layers)
      
      // Different padding for top and bottom to shift network upward
      const topPadding = 10
      const bottomPadding = 40
      const availableHeight = height - topPadding - bottomPadding

      // Draw connections
      for (let i = 0; i < layers.length - 1; i++) {
        const currentLayerSize = layers[i]
        const nextLayerSize = layers[i + 1]
        const currentX = layerSpacing * (i + 1)
        const nextX = layerSpacing * (i + 2)

        for (let j = 0; j < currentLayerSize; j++) {
          const currentY = topPadding + (availableHeight / (currentLayerSize + 1)) * (j + 1)

          for (let k = 0; k < nextLayerSize; k++) {
            const nextY = topPadding + (availableHeight / (nextLayerSize + 1)) * (k + 1)

            const weight = weightsRef.current[i]?.[j]?.[k] ?? 0
            const opacity = Math.abs(weight) * 0.5 + 0.1
            const hue = weight > 0 ? 195 : 320 // cyan for positive, magenta for negative

            ctx.strokeStyle = `oklch(0.7 0.25 ${hue} / ${opacity})`
            ctx.lineWidth = Math.abs(weight) * 2 + 0.5

            ctx.beginPath()
            ctx.moveTo(currentX, currentY)
            ctx.lineTo(nextX, nextY)
            ctx.stroke()

            // Animated pulse effect when training
            if (isTraining && Math.random() > 0.95) {
              ctx.strokeStyle = `oklch(0.9 0.25 ${hue} / 0.8)`
              ctx.lineWidth = 3
              ctx.stroke()
            }
          }
        }
      }

      // Draw neurons
      for (let i = 0; i < layers.length; i++) {
        const layerSize = layers[i]
        const x = layerSpacing * (i + 1)

        for (let j = 0; j < layerSize; j++) {
          const y = topPadding + (availableHeight / (layerSize + 1)) * (j + 1)

          // Neuron glow
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, 15)
          gradient.addColorStop(0, "oklch(0.7 0.25 195 / 0.8)")
          gradient.addColorStop(0.5, "oklch(0.7 0.25 195 / 0.3)")
          gradient.addColorStop(1, "oklch(0.7 0.25 195 / 0)")

          ctx.fillStyle = gradient
          ctx.beginPath()
          ctx.arc(x, y, 15, 0, Math.PI * 2)
          ctx.fill()

          // Neuron core
          ctx.fillStyle = isTraining && Math.random() > 0.7 ? "oklch(0.9 0.25 195)" : "oklch(0.7 0.25 195)"
          ctx.beginPath()
          ctx.arc(x, y, 6, 0, Math.PI * 2)
          ctx.fill()

          // Neuron border
          ctx.strokeStyle = "oklch(0.9 0.25 195 / 0.5)"
          ctx.lineWidth = 2
          ctx.stroke()
        }
      }
    }

    const animate = () => {
      draw()
      if (isTraining) {
        // Simulate weight updates
        if (!weights || weights.length === 0) {
          setVisualWeights((prev) =>
            prev.map((layer) => layer.map((neuron) => neuron.map((w) => w + (Math.random() - 0.5) * 0.01))),
          )
        }
      }
      requestAnimationFrame(animate)
    }

    animate()
  }, [layers, isTraining, weights])

  return (
    <Card className="glass-card glow-cyan p-6 h-[550px]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Network Architecture</h2>
          <p className="text-sm text-muted-foreground">
            {layers.join(" → ")} neurons • {activationFn.toUpperCase()}
          </p>
        </div>
        <div className="flex items-center gap-6">
          {/* Color Legend */}
          <div className="flex flex-col gap-1 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full border" style={{ background: 'oklch(0.7 0.25 195)', border: '1px solid oklch(0.9 0.25 195 / 0.5)' }} />
              <span className="text-muted-foreground">Neurons</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-0.5" style={{ background: 'oklch(0.7 0.25 195 / 0.6)' }} />
              <span className="text-muted-foreground">Positive weights</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-0.5" style={{ background: 'oklch(0.7 0.25 320 / 0.6)' }} />
              <span className="text-muted-foreground">Negative weights</span>
            </div>
          </div>
          
          {isTraining && (
            <motion.div
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1.5, repeat: Number.POSITIVE_INFINITY }}
              className="flex items-center gap-2"
            >
              <div className="w-2 h-2 rounded-full bg-accent" />
              <span className="text-sm text-accent">Training</span>
            </motion.div>
          )}
        </div>
      </div>

      <canvas ref={canvasRef} width={800} height={450} className="w-full h-full" />
    </Card>
  )
}
