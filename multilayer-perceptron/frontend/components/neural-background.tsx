"use client"

import { useEffect, useRef } from "react"

interface FilamentNode {
  x: number
  y: number
  vx: number
  vy: number
  angle: number
  rotationSpeed: number
}

interface ElectricPulse {
  fromIndex: number
  toIndex: number
  progress: number
  speed: number
  color: string
  intensity: number
}

export function NeuralBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const nodesRef = useRef<FilamentNode[]>([])
  const pulsesRef = useRef<ElectricPulse[]>([])
  const animationFrameRef = useRef<number>()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const resizeCanvas = () => {
      if (!canvas) return
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    const nodeCount = 60
    nodesRef.current = Array.from({ length: nodeCount }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      angle: Math.random() * Math.PI * 2,
      rotationSpeed: (Math.random() - 0.5) * 0.02,
    }))

    let time = 0
    const animate = () => {
      if (!canvas || !ctx) return

      ctx.fillStyle = "rgba(15, 23, 42, 0.08)"
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      time += 0.01

      const nodes = nodesRef.current

      nodes.forEach((node) => {
        node.angle += node.rotationSpeed

        const centerX = canvas.width / 2
        const centerY = canvas.height / 2
        const toCenterX = centerX - node.x
        const toCenterY = centerY - node.y
        const distToCenter = Math.sqrt(toCenterX * toCenterX + toCenterY * toCenterY)

        node.vx += (toCenterX / distToCenter) * 0.001
        node.vy += (toCenterY / distToCenter) * 0.001

        node.vx += Math.cos(node.angle) * 0.05
        node.vy += Math.sin(node.angle) * 0.05

        node.vx += (Math.random() - 0.5) * 0.02
        node.vy += (Math.random() - 0.5) * 0.02

        node.vx *= 0.99
        node.vy *= 0.99

        node.x += node.vx
        node.y += node.vy

        if (node.x < 0 || node.x > canvas.width) node.vx *= -0.5
        if (node.y < 0 || node.y > canvas.height) node.vy *= -0.5

        node.x = Math.max(0, Math.min(canvas.width, node.x))
        node.y = Math.max(0, Math.min(canvas.height, node.y))
      })

      if (Math.random() < 0.08) {
        const fromIndex = Math.floor(Math.random() * nodes.length)
        const nearbyNodes = nodes
          .map((node, index) => ({ node, index }))
          .filter((item) => {
            if (item.index === fromIndex) return false
            const dx = item.node.x - nodes[fromIndex].x
            const dy = item.node.y - nodes[fromIndex].y
            const dist = Math.sqrt(dx * dx + dy * dy)
            return dist < 300
          })

        if (nearbyNodes.length > 0) {
          const target = nearbyNodes[Math.floor(Math.random() * nearbyNodes.length)]
          pulsesRef.current.push({
            fromIndex,
            toIndex: target.index,
            progress: 0,
            speed: 0.008 + Math.random() * 0.012,
            color: ["#06b6d4", "#a855f7", "#84cc16"][Math.floor(Math.random() * 3)],
            intensity: 0.4 + Math.random() * 0.3,
          })
        }
      }

      nodes.forEach((node, i) => {
        nodes.slice(i + 1).forEach((other, j) => {
          const dx = other.x - node.x
          const dy = other.y - node.y
          const distance = Math.sqrt(dx * dx + dy * dy)

          if (distance < 280) {
            const opacity = (1 - distance / 280) * 0.15
            const pulse = Math.sin(time * 2 + i * 0.1 + j * 0.1) * 0.3 + 0.7

            ctx.strokeStyle = `rgba(100, 150, 200, ${opacity * pulse})`
            ctx.lineWidth = 0.5 + pulse * 0.3
            ctx.beginPath()
            ctx.moveTo(node.x, node.y)
            ctx.lineTo(other.x, other.y)
            ctx.stroke()
          }
        })
      })

      pulsesRef.current = pulsesRef.current.filter((pulse) => {
        pulse.progress += pulse.speed

        if (pulse.progress <= 1) {
          const fromNode = nodes[pulse.fromIndex]
          const toNode = nodes[pulse.toIndex]

          const x = fromNode.x + (toNode.x - fromNode.x) * pulse.progress
          const y = fromNode.y + (toNode.y - fromNode.y) * pulse.progress

          const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8)
          gradient.addColorStop(
            0,
            `${pulse.color}${Math.floor(pulse.intensity * 180)
              .toString(16)
              .padStart(2, "0")}`,
          )
          gradient.addColorStop(
            0.4,
            `${pulse.color}${Math.floor(pulse.intensity * 100)
              .toString(16)
              .padStart(2, "0")}`,
          )
          gradient.addColorStop(1, `${pulse.color}00`)

          ctx.fillStyle = gradient
          ctx.beginPath()
          ctx.arc(x, y, 8, 0, Math.PI * 2)
          ctx.fill()

          const trailX = fromNode.x + (toNode.x - fromNode.x) * Math.max(0, pulse.progress - 0.15)
          const trailY = fromNode.y + (toNode.y - fromNode.y) * Math.max(0, pulse.progress - 0.15)

          ctx.strokeStyle = `${pulse.color}${Math.floor(pulse.intensity * 100)
            .toString(16)
            .padStart(2, "0")}`
          ctx.lineWidth = 1.5
          ctx.beginPath()
          ctx.moveTo(trailX, trailY)
          ctx.lineTo(x, y)
          ctx.stroke()

          return true
        }
        return false
      })

      nodes.forEach((node) => {
        const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, 3)
        gradient.addColorStop(0, "rgba(150, 180, 220, 0.4)")
        gradient.addColorStop(0.6, "rgba(100, 150, 200, 0.2)")
        gradient.addColorStop(1, "rgba(100, 150, 200, 0)")

        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(node.x, node.y, 3, 0, Math.PI * 2)
        ctx.fill()
      })

      animationFrameRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [])

  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-0" style={{ opacity: 0.6 }} />
}
