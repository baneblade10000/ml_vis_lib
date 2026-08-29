"use client"

import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface TaskSwitcherProps {
  value: "binary" | "multiclass" | "regression"
  onChange: (value: "binary" | "multiclass" | "regression") => void
}

export function TaskSwitcher({ value, onChange }: TaskSwitcherProps) {
  const tasks = [
    { id: "binary" as const, label: "Binary" },
    { id: "multiclass" as const, label: "Multi-class" },
    { id: "regression" as const, label: "Regression" },
  ]

  return (
    <div className="glass-card p-2 flex gap-1 rounded-lg">
      {tasks.map((task) => (
        <button
          key={task.id}
          onClick={() => onChange(task.id)}
          className={cn(
            "relative flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors",
            value === task.id ? "text-foreground" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {value === task.id && (
            <motion.div
              layoutId="task-indicator"
              className="absolute inset-0 bg-primary/20 border border-primary/50 rounded-md"
              transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
            />
          )}
          <span className="relative z-10">{task.label}</span>
        </button>
      ))}
    </div>
  )
}
