import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import {
  DEFAULT_MONTE_CARLO_CONFIG,
  MonteCarloPiEngine,
  PI_TRUE,
} from "@ml-vis/core";
import { ChartBox } from "../useCanvasChart";
import { useMonteCarloT } from "./monteCarloMessages";

const INSIDE_COLOR = "rgba(220, 38, 58, 0.82)";
const INSIDE_GLOW = "rgba(220, 38, 58, 0.35)";
const OUTSIDE_COLOR = "rgba(79, 70, 229, 0.72)";
const OUTSIDE_GLOW = "rgba(79, 70, 229, 0.25)";
const ARC_COLOR = "#4f46e5";
const RECENT_BATCH = 24;
const BASE_TICK_MS = 140;
const MC_PLOT_MARGIN = 8;

function formatPi(value: number, digits = 8): string {
  if (value <= 0) return "—";
  return value.toFixed(digits);
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  engine: MonteCarloPiEngine,
  pulse: number,
): void {
  ctx.clearRect(0, 0, width, height);

  const margin = MC_PLOT_MARGIN;
  const size = Math.min(width, height) - margin * 2;
  const ox = (width - size) / 2;
  const oy = (height - size) / 2;
  const scale = size / 420;
  const mapX = (x: number) => ox + x * size;
  const mapY = (y: number) => oy + (1 - y) * size;

  const bg = ctx.createLinearGradient(ox, oy, ox + size, oy + size);
  bg.addColorStop(0, "#ffffff");
  bg.addColorStop(1, "#f8fafc");
  ctx.fillStyle = bg;
  ctx.fillRect(ox, oy, size, size);

  const cx = mapX(0);
  const cy = mapY(0);

  ctx.save();
  ctx.beginPath();
  ctx.rect(ox, oy, size, size);
  ctx.clip();
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(mapX(1), mapY(0));
  ctx.lineTo(mapX(1), mapY(1));
  ctx.lineTo(mapX(0), mapY(1));
  ctx.arc(cx, cy, size, -Math.PI / 2, 0, false);
  ctx.closePath();
  ctx.fillStyle = "rgba(79, 70, 229, 0.08)";
  ctx.fill();
  ctx.restore();

  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const gx = ox + (size * i) / 4;
    const gy = oy + (size * i) / 4;
    ctx.beginPath();
    ctx.moveTo(gx, oy);
    ctx.lineTo(gx, oy + size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ox, gy);
    ctx.lineTo(ox + size, gy);
    ctx.stroke();
  }

  const points = engine.points;
  const recentStart = Math.max(0, points.length - RECENT_BATCH);
  for (let i = 0; i < points.length; i++) {
    const point = points[i];
    const px = mapX(point.x);
    const py = mapY(point.y);
    const recent = i >= recentStart;
    const fill = point.inside ? INSIDE_COLOR : OUTSIDE_COLOR;

    if (recent) {
      const radius = (2.8 + pulse * 1.2) * scale;

      ctx.beginPath();
      ctx.arc(px, py, radius + 3 * scale, 0, Math.PI * 2);
      ctx.fillStyle = point.inside ? INSIDE_GLOW : OUTSIDE_GLOW;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();

      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.lineWidth = 1;
      ctx.stroke();
    } else {
      const radius = (point.inside ? 1.6 : 1.4) * scale;
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
    }
  }

  const borderWidth = Math.max(1.25, 1.5 * scale);
  ctx.strokeStyle = ARC_COLOR;
  ctx.lineWidth = borderWidth;
  ctx.strokeRect(ox, oy, size, size);

  ctx.beginPath();
  ctx.arc(cx, cy, size, -Math.PI / 2, 0, false);
  ctx.stroke();
}

function MonteCarloCanvas({
  engine,
  pulse,
  insideLabel,
  outsideLabel,
}: {
  engine: MonteCarloPiEngine;
  pulse: number;
  insideLabel: string;
  outsideLabel: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size <= 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawScene(ctx, size, size, engine, pulse);
  }, [engine, pulse, size]);

  return (
    <div className="mc-canvas-wrap">
      <ChartBox aspectRatio={1} onWidth={setSize}>
        {(measuredWidth) => (
          <div
            className="mc-canvas-frame"
            style={{ width: measuredWidth, height: measuredWidth }}
          >
            <canvas
              ref={canvasRef}
              className="mc-canvas"
              style={{ width: measuredWidth, height: measuredWidth, display: "block" }}
            />
            <div className="mc-legend mc-legend--overlay" aria-hidden="true">
              <span className="mc-legend-item mc-legend-item--inside">{insideLabel}</span>
              <span className="mc-legend-item mc-legend-item--outside">{outsideLabel}</span>
            </div>
          </div>
        )}
      </ChartBox>
    </div>
  );
}

export interface MonteCarloPiPlaygroundProps {
  initialSeed?: number;
}

export function MonteCarloPiPlayground({ initialSeed = 42 }: MonteCarloPiPlaygroundProps) {
  const t = useMonteCarloT();
  const engineRef = useRef(
    new MonteCarloPiEngine({ ...DEFAULT_MONTE_CARLO_CONFIG, seed: initialSeed }),
  );
  const [, setTick] = useState(0);
  const bump = useCallback(() => setTick((n) => n + 1), []);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [pulse, setPulse] = useState(0);

  const engine = engineRef.current;

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const intervalMs = Math.max(32, BASE_TICK_MS / speed);
    const timer = window.setInterval(() => {
      frame += 1;
      setPulse(0.5 + 0.5 * Math.sin(frame * 0.12));
      engineRef.current.addBatch();
      bump();
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [playing, speed, bump]);

  const reset = () => {
    engineRef.current.reset();
    setPlaying(false);
    setPulse(0);
    bump();
  };

  const step = () => {
    engineRef.current.addBatch();
    setPulse(1);
    bump();
  };

  return (
    <div className="tf-playground mc-playground">
      <div className="mc-workspace">
        <section className="tf-panel mc-panel--visual">
          <MonteCarloCanvas
            engine={engine}
            pulse={pulse}
            insideLabel={t("inside")}
            outsideLabel={t("outside")}
          />
        </section>

        <aside className="tf-panel mc-panel--sidebar">
          <div className="mc-sidebar-controls">
            <div className="mc-sidebar-actions">
              <button type="button" className="tf-btn tf-btn--ghost" onClick={reset}>
                {t("reset")}
              </button>
              <button
                type="button"
                className={`tf-btn tf-btn--primary${playing ? " playing" : ""}`}
                onClick={() => setPlaying((value) => !value)}
              >
                {playing ? t("pause") : t("play")}
              </button>
              <button type="button" className="tf-btn tf-btn--secondary" onClick={step}>
                {t("step")}
              </button>
            </div>

            <div className="mc-sidebar-fields">
              <div className="tf-toolbar-stat mc-sidebar-stat">
                <span className="label">{t("samples")}</span>
                <span className="value">{engine.totalSamples.toLocaleString()}</span>
              </div>

              <div className="tf-toolbar-field">
                <span className="label">{t("batchSize")}</span>
                <select
                  className="tf-select"
                  value={engine.config.batchSize}
                  onChange={(e) => {
                    engineRef.current.updateConfig({ batchSize: Number(e.target.value) });
                    bump();
                  }}
                >
                  {[4, 8, 16, 32, 64, 128].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>

              <div className="tf-slider mc-speed-slider">
                <label>
                  <span className="tf-slider-header">
                    {t("speed")} <span className="value">{speed}×</span>
                  </span>
                  <input
                    type="range"
                    min={1}
                    max={8}
                    step={1}
                    value={speed}
                    onChange={(e) => setSpeed(Number(e.target.value))}
                    style={
                      { "--range-progress": `${((speed - 1) / 7) * 100}%` } as CSSProperties
                    }
                  />
                </label>
              </div>
            </div>
          </div>

          <div className="mc-stats-stack">
            <div className="mc-pi-display">
              <span className="mc-pi-label">π ≈</span>
              <span className="mc-pi-value">{formatPi(engine.piEstimate, 8)}</span>
            </div>

            <div className="mc-stats-row">
              <div className="tf-stat">
                <span className="tf-stat-label">{t("truePi")}</span>
                <span className="tf-stat-value mc-stat-value--compact">{PI_TRUE.toFixed(6)}</span>
              </div>

              <div className="tf-stat mc-stat--error">
                <span className="tf-stat-label">{t("error")}</span>
                <span className="tf-stat-value mc-stat-value--compact">
                  {engine.totalSamples ? engine.error.toExponential(2) : "—"}
                </span>
              </div>
            </div>

            <div className="mc-stats-row">
              <div className="mc-count mc-count--inside">
                <span>{t("inside")}</span>
                <strong>{engine.insideCount.toLocaleString()}</strong>
              </div>

              <div className="mc-count mc-count--outside">
                <span>{t("outside")}</span>
                <strong>{(engine.totalSamples - engine.insideCount).toLocaleString()}</strong>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
