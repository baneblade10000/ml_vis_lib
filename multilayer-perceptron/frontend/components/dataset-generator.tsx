"use client"

import { useState, useEffect, useCallback } from "react"
import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Sparkles, RefreshCw } from "lucide-react"

export interface DataPoint {
  x: number
  y: number
  label: number
}

interface DatasetGeneratorProps {
  onDatasetGenerated?: (data: DataPoint[]) => void
  decisionBoundary?: {
    cells: { x: number; y: number; value: number }[]
    step: number
  }
}

const interpolateColor = (value: number) => {
  const clamped = Math.max(0, Math.min(1, value))
  const colorA = [6, 182, 212]
  const colorB = [236, 72, 153]

  const mixed = colorA.map((channel, index) => {
    const target = colorB[index]
    return Math.round(channel + (target - channel) * clamped)
  })

  return `rgb(${mixed[0]}, ${mixed[1]}, ${mixed[2]})`
}

export function DatasetGenerator({ onDatasetGenerated, decisionBoundary }: DatasetGeneratorProps) {
  const [datasetType, setDatasetType] = useState<"spiral" | "circular" | "xor" | "moons">("spiral")
  const [sampleSize, setSampleSize] = useState(200)
  const [noiseLevel, setNoiseLevel] = useState(0.1)
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([])

  const generateSpiral = useCallback((samples: number, noise: number): DataPoint[] => {
    const points: DataPoint[] = []
    const n = Math.floor(samples / 2)

    for (let i = 0; i < n; i++) {
      const r = (i / n) * 5
      const t = (i / n) * 4 * Math.PI

      points.push({
        x: r * Math.cos(t) + (Math.random() - 0.5) * noise,
        y: r * Math.sin(t) + (Math.random() - 0.5) * noise,
        label: 0,
      })

      points.push({
        x: r * Math.cos(t + Math.PI) + (Math.random() - 0.5) * noise,
        y: r * Math.sin(t + Math.PI) + (Math.random() - 0.5) * noise,
        label: 1,
      })
    }

    return points
  }, [])

  const generateCircular = useCallback((samples: number, noise: number): DataPoint[] => {
    const points: DataPoint[] = []
    const n = Math.floor(samples / 2)

    for (let i = 0; i < n; i++) {
      const angle = Math.random() * 2 * Math.PI

      const r1 = 2 + (Math.random() - 0.5) * noise
      points.push({
        x: r1 * Math.cos(angle),
        y: r1 * Math.sin(angle),
        label: 0,
      })

      const r2 = 5 + (Math.random() - 0.5) * noise
      points.push({
        x: r2 * Math.cos(angle),
        y: r2 * Math.sin(angle),
        label: 1,
      })
    }

    return points
  }, [])

  const generateXOR = useCallback((samples: number, noise: number): DataPoint[] => {
    const points: DataPoint[] = []
    const n = Math.floor(samples / 4)

    for (let i = 0; i < n; i++) {
      points.push({
        x: Math.random() * 4 + 1 + (Math.random() - 0.5) * noise,
        y: Math.random() * 4 + 1 + (Math.random() - 0.5) * noise,
        label: 0,
      })

      points.push({
        x: Math.random() * 4 - 5 + (Math.random() - 0.5) * noise,
        y: Math.random() * 4 + 1 + (Math.random() - 0.5) * noise,
        label: 1,
      })

      points.push({
        x: Math.random() * 4 - 5 + (Math.random() - 0.5) * noise,
        y: Math.random() * 4 - 5 + (Math.random() - 0.5) * noise,
        label: 0,
      })

      points.push({
        x: Math.random() * 4 + 1 + (Math.random() - 0.5) * noise,
        y: Math.random() * 4 - 5 + (Math.random() - 0.5) * noise,
        label: 1,
      })
    }

    return points
  }, [])

  const generateMoons = useCallback((samples: number, noise: number): DataPoint[] => {
    const points: DataPoint[] = []
    const n = Math.floor(samples / 2)
    const radius = 3 // Larger radius for better visibility

    for (let i = 0; i < n; i++) {
      // First moon (upper)
      const t1 = Math.PI * i / n
      points.push({
        x: radius * Math.cos(t1) + (Math.random() - 0.5) * noise,
        y: radius * Math.sin(t1) + (Math.random() - 0.5) * noise,
        label: 0,
      })

      // Second moon (lower, inverted)
      const t2 = Math.PI * i / n
      points.push({
        x: radius - radius * Math.cos(t2) + (Math.random() - 0.5) * noise,
        y: -radius * Math.sin(t2) - 1.5 + (Math.random() - 0.5) * noise,
        label: 1,
      })
    }

    return points
  }, [])

  const generateDataset = useCallback(() => {
    let points: DataPoint[] = []

    switch (datasetType) {
      case "spiral":
        points = generateSpiral(sampleSize, noiseLevel)
        break
      case "circular":
        points = generateCircular(sampleSize, noiseLevel)
        break
      case "xor":
        points = generateXOR(sampleSize, noiseLevel)
        break
      case "moons":
        points = generateMoons(sampleSize, noiseLevel)
        break
    }

    setDataPoints(points)
    if (onDatasetGenerated) {
      onDatasetGenerated(points)
    }
  }, [datasetType, sampleSize, noiseLevel, generateSpiral, generateCircular, generateXOR, generateMoons, onDatasetGenerated])

  useEffect(() => {
    generateDataset()
  }, [generateDataset])

  return (
    <Card className="glass-panel p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-primary" />
          <h3 className="font-semibold text-foreground">Dataset Generator</h3>
        </div>
        <Button size="sm" variant="ghost" onClick={generateDataset} className="h-8 w-8 p-0">
          <RefreshCw className="w-4 h-4" />
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Left side - Configuration */}
        <div className="space-y-6">
          <div className="space-y-3">
            <Label className="text-sm text-muted-foreground">Pattern Type</Label>
            <div className="grid grid-cols-2 gap-2">
              {(["spiral", "circular", "xor", "moons"] as const).map((type) => (
                <Button
                  key={type}
                  variant={datasetType === type ? "default" : "outline"}
                  size="sm"
                  onClick={() => setDatasetType(type)}
                  className="capitalize"
                >
                  {type === "moons" ? "two moons" : type}
                </Button>
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <Label className="text-sm text-muted-foreground">Sample Size</Label>
            <div className="flex gap-2">
              <Input
                type="number"
                value={sampleSize}
                onChange={(e) => setSampleSize(Math.max(10, Math.min(1000, Number.parseInt(e.target.value) || 50)))}
                min={10}
                max={1000}
                className="flex-1"
                placeholder="50-1000"
              />
            </div>
            <Slider
              value={[sampleSize]}
              onValueChange={([value]) => setSampleSize(value)}
              min={50}
              max={500}
              step={50}
              className="w-full"
            />
          </div>

          <div className="space-y-3">
            <Label className="text-sm text-muted-foreground">Noise Level</Label>
            <div className="flex gap-2">
              <Input
                type="number"
                value={noiseLevel}
                onChange={(e) => setNoiseLevel(Math.max(0, Math.min(1, Number.parseFloat(e.target.value) || 0)))}
                min={0}
                max={1}
                step={0.01}
                className="flex-1"
                placeholder="0.00-1.00"
              />
            </div>
            <Slider
              value={[noiseLevel * 100]}
              onValueChange={([value]) => setNoiseLevel(value / 100)}
              min={0}
              max={50}
              step={5}
              className="w-full"
            />
          </div>

          <div className="text-xs text-muted-foreground">
            {dataPoints.length} points generated
          </div>
        </div>

        {/* Right side - Visualization */}
        <div className="relative aspect-square bg-background/50 rounded-lg border border-border/50 overflow-hidden">
          <svg viewBox="-6 -6 12 12" className="w-full h-full">
            {decisionBoundary?.cells.map((cell, index) => (
              <rect
                key={`cell-${index}`}
                x={cell.x - (decisionBoundary.step / 2)}
                y={-(cell.y + decisionBoundary.step / 2)}
                width={decisionBoundary.step}
                height={decisionBoundary.step}
                fill={interpolateColor(cell.value)}
                opacity={0.3}
              />
            ))}
            <line x1="-6" y1="0" x2="6" y2="0" stroke="currentColor" strokeWidth="0.02" opacity="0.2" />
            <line x1="0" y1="-6" x2="0" y2="6" stroke="currentColor" strokeWidth="0.02" opacity="0.2" />

            {dataPoints.map((point, i) => (
              <motion.circle
                key={`${i}-${point.x}-${point.y}`}
                cx={point.x}
                cy={-point.y}
                r="0.08"
                fill={point.label === 0 ? "rgb(6, 182, 212)" : "rgb(236, 72, 153)"}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 0.8 }}
                transition={{ delay: i * 0.001, duration: 0.3 }}
              />
            ))}
          </svg>
        </div>
      </div>
    </Card>
  )
}
