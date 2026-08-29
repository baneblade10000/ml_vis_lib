"use client"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Plus, Minus, Layers } from "lucide-react"
import { motion } from "framer-motion"

interface LayerControlPanelProps {
  layers: number[]
  onLayersChange: (layers: number[]) => void
  activationFn: string
  onActivationChange: (fn: string) => void
}

export function LayerControlPanel({
  layers,
  onLayersChange,
  activationFn,
  onActivationChange,
}: LayerControlPanelProps) {
  // Maximum 5 hidden layers (input + 5 hidden + output = 7 total)
  const maxHiddenLayers = 5
  const maxNeuronsPerLayer = 12
  
  const addLayer = () => {
    // Check if we can add more hidden layers
    const currentHiddenLayers = layers.length - 2 // exclude input and output
    if (currentHiddenLayers >= maxHiddenLayers) return
    
    const newLayers = [...layers]
    newLayers.splice(layers.length - 1, 0, 8)
    onLayersChange(newLayers)
  }

  const removeLayer = () => {
    if (layers.length > 2) {
      const newLayers = [...layers]
      newLayers.splice(layers.length - 2, 1)
      onLayersChange(newLayers)
    }
  }

  const updateNeuronCount = (index: number, value: number) => {
    const newLayers = [...layers]
    newLayers[index] = Math.max(1, Math.min(maxNeuronsPerLayer, value))
    onLayersChange(newLayers)
  }

  return (
    <Card className="glass-card p-6">
      <div className="flex items-center gap-2 mb-4">
        <Layers className="w-5 h-5 text-primary" />
        <div>
          <h3 className="text-lg font-semibold text-foreground">Layer Control</h3>
          <p className="text-xs text-muted-foreground">Max {maxHiddenLayers} hidden layers • {maxNeuronsPerLayer} neurons per layer</p>
        </div>
      </div>

      <div className="space-y-4">
        {/* Layer Configuration */}
        <div className="space-y-3">
          {layers.map((neurons, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="space-y-2"
            >
              <Label className="text-sm text-muted-foreground">
                {index === 0 ? "Input" : index === layers.length - 1 ? "Output" : `Hidden ${index}`}
              </Label>
              <Input
                type="number"
                value={neurons}
                onChange={(e) => updateNeuronCount(index, Number.parseInt(e.target.value) || 1)}
                min={1}
                max={maxNeuronsPerLayer}
                className="bg-muted/50 border-border/50"
                disabled={index === 0 || index === layers.length - 1}
              />
            </motion.div>
          ))}
        </div>

        {/* Add/Remove Layer Buttons */}
        <div className="flex gap-2">
          <Button
            onClick={addLayer}
            variant="outline"
            size="sm"
            className="flex-1 border-primary/30 hover:bg-primary/10 hover:border-primary/50 bg-transparent"
            disabled={layers.length - 2 >= maxHiddenLayers}
          >
            <Plus className="w-4 h-4 mr-1" />
            Add Layer
          </Button>
          <Button
            onClick={removeLayer}
            variant="outline"
            size="sm"
            className="flex-1 border-destructive/30 hover:bg-destructive/10 hover:border-destructive/50 bg-transparent"
            disabled={layers.length <= 2}
          >
            <Minus className="w-4 h-4 mr-1" />
            Remove
          </Button>
        </div>

        {/* Activation Function */}
        <div className="space-y-2 pt-4 border-t border-border/50">
          <Label className="text-sm text-muted-foreground">Activation Function</Label>
          <Select value={activationFn} onValueChange={onActivationChange}>
            <SelectTrigger className="bg-muted/50 border-border/50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="relu">ReLU</SelectItem>
              <SelectItem value="sigmoid">Sigmoid</SelectItem>
              <SelectItem value="tanh">Tanh</SelectItem>
              <SelectItem value="leaky_relu">Leaky ReLU</SelectItem>
              <SelectItem value="elu">ELU</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
    </Card>
  )
}
