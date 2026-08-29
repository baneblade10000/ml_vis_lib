"use client"

import { useMemo, useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Play, RotateCcw } from "lucide-react"
import { Line } from "react-chartjs-2"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js"

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

interface TrainingDashboardProps {
  isTraining: boolean
  canTrain: boolean
  history: {
    loss: number[]
    accuracy: number[]
    valLoss: (number | null)[]
    valAccuracy: (number | null)[]
  } | null
  metrics: {
    loss: number
    accuracy?: number
    precision?: number
    recall?: number
    f1Score?: number
    mse?: number
  } | null
  onStartTraining: (config: { 
    epochs: number; 
    batchSize: number; 
    validationSplit: number;
    enableEarlyStopping: boolean;
    earlyStoppingPatience: number;
    earlyStoppingMinDelta: number;
  }) => void
  onReset: () => void
  error?: string | null
}

export function TrainingDashboard({
  isTraining,
  canTrain,
  history,
  metrics,
  onStartTraining,
  onReset,
  error,
}: TrainingDashboardProps) {
  const [epochs, setEpochs] = useState(100)
  const [batchSize, setBatchSize] = useState(32)
  const [validationSplit, setValidationSplit] = useState(0.2)
  const [enableEarlyStopping, setEnableEarlyStopping] = useState(false)
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(20)
  const [earlyStoppingMinDelta, setEarlyStoppingMinDelta] = useState(0.05)

  const currentEpoch = history?.loss.length ?? 0
  const latestLoss = history?.loss.at(-1) ?? 0
  const latestAccuracy = history?.accuracy.at(-1) ?? 0
  const latestValLoss = history?.valLoss.at(-1) ?? null
  const latestValAccuracy = history?.valAccuracy.at(-1) ?? null

  const chartData = useMemo(() => {
    const labels = history ? history.loss.map((_, index) => (index + 1).toString()) : []
    const datasets: any[] = []

    if (history) {
      datasets.push({
        label: "Training Loss",
        data: history.loss,
        borderColor: "oklch(0.7 0.25 195)",
        backgroundColor: "oklch(0.7 0.25 195 / 0.1)",
        fill: true,
        tension: 0.4,
        pointRadius: 0,
      })

      if (history.valLoss.some((value) => value !== null)) {
        datasets.push({
          label: "Validation Loss",
          data: history.valLoss.map((value) => value ?? NaN),
          borderColor: "oklch(0.75 0.2 20)",
          backgroundColor: "oklch(0.75 0.2 20 / 0.1)",
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          spanGaps: true,
        })
      }
    }

    return { labels, datasets }
  }, [history])

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        grid: {
          color: "oklch(0.25 0.03 265 / 0.2)",
        },
        ticks: {
          color: "oklch(0.65 0.02 265)",
        },
      },
    },
  }

  return (
    <Card className="glass-card glow-magenta p-6">
      <h3 className="text-lg font-semibold text-foreground mb-4">Training Metrics</h3>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">Epoch</p>
          <p className="text-2xl font-bold text-foreground">{currentEpoch}</p>
        </div>
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">Loss</p>
          <p className="text-2xl font-bold text-primary">{latestLoss.toFixed(4)}</p>
        </div>
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">Accuracy</p>
          <p className="text-2xl font-bold text-accent">{(latestAccuracy * 100).toFixed(2)}%</p>
        </div>
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">Validation Loss</p>
          <p className="text-2xl font-bold text-secondary">
            {latestValLoss !== null && Number.isFinite(latestValLoss) ? latestValLoss.toFixed(4) : "—"}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="space-y-2">
          <Label htmlFor="epochs" className="text-sm text-muted-foreground">
            Epochs
          </Label>
          <Input
            id="epochs"
            type="number"
            min={1}
            value={epochs}
            onChange={(event) => setEpochs(Math.max(1, Number.parseInt(event.target.value) || 1))}
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="batch-size" className="text-sm text-muted-foreground">
            Batch Size
          </Label>
          <Input
            id="batch-size"
            type="number"
            min={1}
            value={batchSize}
            onChange={(event) => setBatchSize(Math.max(1, Number.parseInt(event.target.value) || 1))}
          />
        </div>
        <div className="space-y-2">
          <Label className="text-sm text-muted-foreground">
            Validation Split ({(validationSplit * 100).toFixed(0)}%)
          </Label>
          <Slider
            value={[validationSplit * 100]}
            onValueChange={([value]) => setValidationSplit(Math.round(value) / 100)}
            min={0}
            max={40}
            step={5}
          />
        </div>
        
        {/* Early Stopping Configuration */}
        <div className="space-y-4 border-t pt-4">
          <div className="flex items-center justify-between">
            <Label className="text-sm text-muted-foreground">Early Stopping</Label>
            <Switch
              checked={enableEarlyStopping}
              onCheckedChange={setEnableEarlyStopping}
            />
          </div>
          
          {enableEarlyStopping && (
            <>
              <div className="space-y-2">
                <Label className="text-sm text-muted-foreground">Patience (epochs)</Label>
                <Input
                  type="number"
                  value={earlyStoppingPatience}
                  onChange={(event) => setEarlyStoppingPatience(Math.max(1, Number.parseInt(event.target.value) || 20))}
                  min={1}
                  max={100}
                  placeholder="20"
                />
              </div>
              
              <div className="space-y-2">
                <Label className="text-sm text-muted-foreground">Min Delta</Label>
                <Input
                  type="number"
                  value={earlyStoppingMinDelta}
                  onChange={(event) => setEarlyStoppingMinDelta(Math.max(0, Number.parseFloat(event.target.value) || 0.05))}
                  min={0}
                  max={1}
                  step={0.05}
                  placeholder="0.05"
                />
              </div>
            </>
          )}
        </div>
      </div>

      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 text-sm">
          <div className="space-y-1">
            <p className="text-muted-foreground">Final Loss</p>
            <p className="text-foreground font-medium">{metrics.loss.toFixed(4)}</p>
          </div>
          {metrics.accuracy !== undefined && (
            <div className="space-y-1">
              <p className="text-muted-foreground">Final Accuracy</p>
              <p className="text-foreground font-medium">{(metrics.accuracy * 100).toFixed(2)}%</p>
            </div>
          )}
          {metrics.precision !== undefined && (
            <div className="space-y-1">
              <p className="text-muted-foreground">Precision</p>
              <p className="text-foreground font-medium">{(metrics.precision * 100).toFixed(2)}%</p>
            </div>
          )}
          {metrics.recall !== undefined && (
            <div className="space-y-1">
              <p className="text-muted-foreground">Recall</p>
              <p className="text-foreground font-medium">{(metrics.recall * 100).toFixed(2)}%</p>
            </div>
          )}
          {metrics.f1Score !== undefined && (
            <div className="space-y-1">
              <p className="text-muted-foreground">F1 Score</p>
              <p className="text-foreground font-medium">{(metrics.f1Score * 100).toFixed(2)}%</p>
            </div>
          )}
          {metrics.mse !== undefined && (
            <div className="space-y-1">
              <p className="text-muted-foreground">MSE</p>
              <p className="text-foreground font-medium">{metrics.mse.toFixed(4)}</p>
            </div>
          )}
          {latestValAccuracy !== null && Number.isFinite(latestValAccuracy) && (
            <div className="space-y-1">
              <p className="text-muted-foreground">Validation Accuracy</p>
              <p className="text-foreground font-medium">{(latestValAccuracy * 100).toFixed(2)}%</p>
            </div>
          )}
        </div>
      )}

      <div className="h-40 mb-6">
        {history && history.loss.length > 0 ? (
          <Line data={chartData} options={chartOptions} />
        ) : (
          <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
            Train the model to visualize loss curves
          </div>
        )}
      </div>

      {error ? (
        <p className="text-sm text-red-400 mb-4">{error}</p>
      ) : (
        !canTrain && (
          <p className="text-sm text-muted-foreground mb-4">
            Generate a dataset and ensure the WASM module has loaded to enable training.
          </p>
        )
      )}

      <div className="flex gap-2">
        <Button
          onClick={() => onStartTraining({ 
            epochs, 
            batchSize, 
            validationSplit, 
            enableEarlyStopping, 
            earlyStoppingPatience, 
            earlyStoppingMinDelta 
          })}
          className="flex-1 bg-primary hover:bg-primary/90 text-primary-foreground"
          disabled={!canTrain || isTraining}
        >
          <Play className="w-4 h-4 mr-2" />
          {isTraining ? "Training..." : "Train"}
        </Button>
        <Button
          onClick={() => {
            onReset()
            setEpochs(100)
            setBatchSize(32)
            setValidationSplit(0.2)
            setEnableEarlyStopping(false)
            setEarlyStoppingPatience(5)
            setEarlyStoppingMinDelta(0.001)
          }}
          variant="outline"
          className="border-border/50 hover:bg-muted/50 bg-transparent"
          disabled={isTraining}
        >
          <RotateCcw className="w-4 h-4" />
        </Button>
      </div>
    </Card>
  )
}
