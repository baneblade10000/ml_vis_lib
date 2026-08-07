import { useEffect, useRef } from "react";
import { inferYDomain, renderCurve, type LossHistoryPoint } from "@ml-vis/core";

const TRAIN_STROKE = "rgb(148, 163, 184)"; // --nn-text-muted
const TEST_STROKE = "rgb(15, 23, 42)"; // --nn-text
const CHART_CSS_HEIGHT = 72;

/** Expand toward new peaks quickly; shrink unused headroom slowly. */
const DOMAIN_EXPAND = 0.4;
const DOMAIN_SHRINK = 0.05;
const DOMAIN_EPS = 1e-4;

export interface NetworkLossChartProps {
  history: LossHistoryPoint[];
  title: string;
  trainLabel: string;
  testLabel: string;
  /** Live scalars shown under the chart. */
  lossTest: number;
  lossTrain: number;
}

function easeDomain(current: [number, number], target: [number, number]): [number, number] {
  const mix = (c: number, t: number) => {
    const a = t > c ? DOMAIN_EXPAND : DOMAIN_SHRINK;
    return c + (t - c) * a;
  };
  return [mix(current[0], target[0]), mix(current[1], target[1])];
}

function domainClose(a: [number, number], b: [number, number]): boolean {
  return Math.abs(a[0] - b[0]) < DOMAIN_EPS && Math.abs(a[1] - b[1]) < DOMAIN_EPS;
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
  const historyRef = useRef(history);
  const displayDomainRef = useRef<[number, number] | null>(null);
  const targetDomainRef = useRef<[number, number]>([0, 1]);
  const rafRef = useRef(0);
  const testValRef = useRef<HTMLSpanElement>(null);
  const trainValRef = useRef<HTMLSpanElement>(null);

  historyRef.current = history;

  const schedulePaint = useRef(() => {});

  useEffect(() => {
    if (testValRef.current) testValRef.current.textContent = lossTest.toFixed(3);
    if (trainValRef.current) trainValRef.current.textContent = lossTrain.toFixed(3);
  }, [lossTest, lossTrain]);

  // Stable paint loop — ResizeObserver must not restart on every history tick.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const paintFrame = () => {
      rafRef.current = 0;
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

      const hist = historyRef.current;
      if (hist.length < 1) return;

      const series = hist.length === 1 ? [hist[0]!, hist[0]!] : hist;
      const train = series.map((p) => p.train);
      const test = series.map((p) => p.test);
      const [y0, y1] = inferYDomain([...train, ...test], 0.12);
      const target: [number, number] = [Math.max(0, y0), Math.max(y1, 1e-3)];
      targetDomainRef.current = target;

      let domain = displayDomainRef.current;
      if (!domain) {
        domain = target;
        displayDomainRef.current = domain;
      } else {
        domain = easeDomain(domain, target);
        displayDomainRef.current = domain;
      }

      renderCurve(canvas, train, {
        yDomain: domain,
        stroke: TRAIN_STROKE,
        fill: false,
        baseline: false,
        clear: true,
        lineWidth: Math.max(1.25 * dpr, w / 180),
      });
      renderCurve(canvas, test, {
        yDomain: domain,
        stroke: TEST_STROKE,
        fill: false,
        baseline: false,
        clear: false,
        lineWidth: Math.max(1.5 * dpr, w / 160),
      });

      if (!domainClose(domain, targetDomainRef.current)) {
        rafRef.current = requestAnimationFrame(paintFrame);
      }
    };

    const schedule = () => {
      if (rafRef.current) return;
      rafRef.current = requestAnimationFrame(paintFrame);
    };
    schedulePaint.current = schedule;

    schedule();
    const ro = new ResizeObserver(() => {
      displayDomainRef.current = null;
      schedule();
    });
    ro.observe(canvas);
    return () => {
      ro.disconnect();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    };
  }, []);

  useEffect(() => {
    // Reset / short histories: snap scale instead of easing from the old run.
    if (history.length <= 2) displayDomainRef.current = null;
    schedulePaint.current();
  }, [history]);

  return (
    <div className="nn-loss-chart-wrap">
      <div className="nn-loss-chart-label">
        <span>{title}</span>
        <div className="nn-loss-chart-legend">
          <span className="legend-train">{trainLabel}</span>
          <span className="legend-test">{testLabel}</span>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        className="nn-linechart"
        role="img"
        aria-label={`${title}: ${trainLabel} ${lossTrain.toFixed(3)}, ${testLabel} ${lossTest.toFixed(3)}`}
      />
      <div className="nn-loss-chart-values">
        <span ref={testValRef} className="nn-loss-chart-values__test">
          {lossTest.toFixed(3)}
        </span>
        <span className="nn-loss-chart-values__sep">/</span>
        <span ref={trainValRef} className="nn-loss-chart-values__train">
          {lossTrain.toFixed(3)}
        </span>
      </div>
    </div>
  );
}
