import { useEffect, useRef } from "react";
import { inferYDomain, renderCurve, type LossHistoryPoint } from "@ml-vis/core";

const TRAIN_STROKE = "rgb(148, 163, 184)"; // --tf-text-muted
const TEST_STROKE = "rgb(15, 23, 42)"; // --tf-text
const CHART_CSS_HEIGHT = 72;

export interface NetworkLossChartProps {
  history: LossHistoryPoint[];
  title: string;
  trainLabel: string;
  testLabel: string;
  /** Live scalars shown under the chart. */
  lossTest: number;
  lossTrain: number;
}

export function NetworkLossChart({
  history,
  title,
  trainLabel,
  testLabel,
  lossTest,
  lossTrain,
}: NetworkLossChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const paint = () => {
      const cssW = Math.max(1, Math.floor(canvas.clientWidth));
      const cssH = CHART_CSS_HEIGHT;
      const dpr = window.devicePixelRatio || 1;
      const w = Math.round(cssW * dpr);
      const h = Math.round(cssH * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, w, h);

      if (history.length < 1) return;

      // Duplicate a lone point so renderCurve (n >= 2) can draw a flat segment.
      const series = history.length === 1 ? [history[0]!, history[0]!] : history;
      const train = series.map((p) => p.train);
      const test = series.map((p) => p.test);
      const [y0, y1] = inferYDomain([...train, ...test], 0.12);
      // Loss is non-negative — don't leave empty space below zero.
      const yDomain: [number, number] = [Math.max(0, y0), Math.max(y1, 1e-3)];

      renderCurve(canvas, train, {
        yDomain,
        stroke: TRAIN_STROKE,
        fill: false,
        baseline: false,
        clear: true,
        lineWidth: Math.max(1.25 * dpr, w / 180),
      });
      renderCurve(canvas, test, {
        yDomain,
        stroke: TEST_STROKE,
        fill: false,
        baseline: false,
        clear: false,
        lineWidth: Math.max(1.5 * dpr, w / 160),
      });
    };

    paint();
    const ro = new ResizeObserver(() => paint());
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [history]);

  return (
    <div className="tf-loss-chart-wrap">
      <div className="tf-loss-chart-label">
        <span>{title}</span>
        <div className="tf-loss-chart-legend">
          <span className="legend-train">{trainLabel}</span>
          <span className="legend-test">{testLabel}</span>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        className="tf-linechart"
        role="img"
        aria-label={`${title}: ${trainLabel} ${lossTrain.toFixed(3)}, ${testLabel} ${lossTest.toFixed(3)}`}
      />
      <div className="tf-loss-chart-values">
        <span className="tf-loss-chart-values__test">{lossTest.toFixed(3)}</span>
        <span className="tf-loss-chart-values__sep">/</span>
        <span className="tf-loss-chart-values__train">{lossTrain.toFixed(3)}</span>
      </div>
    </div>
  );
}
