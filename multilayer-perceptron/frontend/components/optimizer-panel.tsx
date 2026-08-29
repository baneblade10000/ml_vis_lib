"use client"

import { Card } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Settings2, TrendingUp } from "lucide-react"

type OptimizerType = "gd" | "sgd" | "sgd_momentum" | "rmsprop" | "adam"
type RegularizationType = "none" | "l1" | "l2" | "elastic_net"

interface OptimizerPanelProps {
  optimizer: OptimizerType
  onOptimizerChange: (type: OptimizerType) => void
  learningRate: number
  onLearningRateChange: (value: number) => void
  regularization: RegularizationType
  onRegularizationChange: (type: RegularizationType) => void
  l1Lambda: number
  onL1LambdaChange: (value: number) => void
  l2Lambda: number
  onL2LambdaChange: (value: number) => void
}

export function OptimizerPanel({
  optimizer,
  onOptimizerChange,
  learningRate,
  onLearningRateChange,
  regularization,
  onRegularizationChange,
  l1Lambda,
  onL1LambdaChange,
  l2Lambda,
  onL2LambdaChange,
}: OptimizerPanelProps) {
  return (
    <Card className="glass-card p-6">
      <div className="flex items-center gap-2 mb-6">
        <Settings2 className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold text-foreground">Optimizer & Regularization</h3>
      </div>

      <div className="space-y-6">
        {/* Optimizer Section */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-primary" />
            <h4 className="font-medium text-foreground">Optimization Algorithm</h4>
          </div>
          
          <div className="space-y-2">
            <Label className="text-sm text-muted-foreground">Algorithm</Label>
            <Select value={optimizer} onValueChange={onOptimizerChange}>
              <SelectTrigger className="bg-muted/50 border-border/50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gd">Gradient Descent</SelectItem>
                <SelectItem value="sgd">SGD</SelectItem>
                <SelectItem value="sgd_momentum">SGD with Momentum</SelectItem>
                <SelectItem value="rmsprop">RMSProp</SelectItem>
                <SelectItem value="adam">Adam</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label className="text-sm text-muted-foreground">Learning Rate</Label>
            <div className="flex gap-2">
              <Input
                type="number"
                value={learningRate}
                onChange={(e) => onLearningRateChange(Math.max(0.0001, Math.min(1, Number.parseFloat(e.target.value) || 0.01)))}
                min={0.0001}
                max={1}
                step={0.0001}
                className="flex-1"
                placeholder="0.0001-1.0"
              />
            </div>
            <Slider
              value={[learningRate * 1000]}
              onValueChange={([value]) => onLearningRateChange(value / 1000)}
              min={0.1}
              max={100}
              step={0.1}
              className="w-full"
            />
          </div>
        </div>

        {/* Regularization Section */}
        <div className="space-y-4 pt-4 border-t border-border/50">
          <h4 className="font-medium text-foreground">Regularization</h4>
          
          <div className="space-y-2">
            <Label className="text-sm text-muted-foreground">Type</Label>
            <Select value={regularization} onValueChange={onRegularizationChange}>
              <SelectTrigger className="bg-muted/50 border-border/50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="l1">L1 (Lasso)</SelectItem>
                <SelectItem value="l2">L2 (Ridge)</SelectItem>
                <SelectItem value="elastic_net">Elastic Net (L1 + L2)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* L1 Lambda */}
          {(regularization === "l1" || regularization === "elastic_net") && (
            <div className="space-y-2">
              <Label className="text-sm text-muted-foreground">L1 Lambda (α)</Label>
              <div className="flex gap-2">
                <Input
                  type="number"
                  value={l1Lambda}
                  onChange={(e) => onL1LambdaChange(Math.max(0, Math.min(1, Number.parseFloat(e.target.value) || 0)))}
                  min={0}
                  max={1}
                  step={0.001}
                  className="flex-1"
                  placeholder="0.000-1.000"
                />
              </div>
              <Slider
                value={[l1Lambda * 1000]}
                onValueChange={([value]) => onL1LambdaChange(value / 1000)}
                min={0}
                max={100}
                step={1}
                className="w-full"
              />
            </div>
          )}

          {/* L2 Lambda */}
          {(regularization === "l2" || regularization === "elastic_net") && (
            <div className="space-y-2">
              <Label className="text-sm text-muted-foreground">L2 Lambda (λ)</Label>
              <div className="flex gap-2">
                <Input
                  type="number"
                  value={l2Lambda}
                  onChange={(e) => onL2LambdaChange(Math.max(0, Math.min(1, Number.parseFloat(e.target.value) || 0)))}
                  min={0}
                  max={1}
                  step={0.001}
                  className="flex-1"
                  placeholder="0.000-1.000"
                />
              </div>
              <Slider
                value={[l2Lambda * 1000]}
                onValueChange={([value]) => onL2LambdaChange(value / 1000)}
                min={0}
                max={100}
                step={1}
                className="w-full"
              />
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}
