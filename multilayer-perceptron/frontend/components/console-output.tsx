"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Terminal } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

interface ConsoleOutputProps {
  isTraining: boolean
  trainingLogs?: string[]
  logs: string[]
}

export function ConsoleOutput({ isTraining, trainingLogs = [], logs: externalLogs }: ConsoleOutputProps) {
  const [logs, setLogs] = useState<string[]>(["> Neural network initialized", "> Ready to train"])

  // Capture console.log messages from WASM
  useEffect(() => {
    const originalLog = console.log
    
    console.log = (...args: any[]) => {
      const message = args.map(arg => typeof arg === 'string' ? arg : JSON.stringify(arg)).join(' ')
      const timestamp = new Date().toLocaleTimeString()
      
      // Check if this looks like a training message (contains "Epoch" or training-related keywords)
      if (message.includes('Epoch') || message.includes('loss:') || message.includes('Training') || message.includes('accuracy:')) {
        setLogs(prev => [...prev.slice(-15), `[${timestamp}] ${message}`])
      }
      
      // Call original console.log for browser console
      originalLog.apply(console, args)
    }

    return () => {
      console.log = originalLog
    }
  }, [])

  // Add external training logs if provided
  useEffect(() => {
    if (trainingLogs.length > 0) {
      const timestamp = new Date().toLocaleTimeString()
      const newLogs = trainingLogs.map(log => `[${timestamp}] ${log}`)
      setLogs(prev => [...prev, ...newLogs].slice(-15))
    }
  }, [trainingLogs])

  // Sync with external logs from useMLTraining hook
  useEffect(() => {
    if (externalLogs && externalLogs.length > 0) {
      setLogs(externalLogs)
    }
  }, [externalLogs])

  // Add initial training message
  useEffect(() => {
    if (isTraining && trainingLogs.length === 0) {
      const timestamp = new Date().toLocaleTimeString()
      setLogs(prev => [...prev.slice(-14), `[${timestamp}] Training started...`])
    }
  }, [isTraining, trainingLogs.length])

  const visibleLogs = logs.slice(-50)

  return (
    <Card className="glass-card p-6">
      <div className="flex items-center gap-2 mb-4">
        <Terminal className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold text-foreground">Console</h3>
      </div>

      <div className="bg-muted/30 rounded-lg p-4 h-48 overflow-y-auto font-mono text-xs space-y-1">
        <AnimatePresence initial={false}>
          {visibleLogs.map((log: string, index: number) => (
            <motion.div
              key={`${log}-${index}`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              className="text-muted-foreground"
            >
              {log}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </Card>
  )
}
