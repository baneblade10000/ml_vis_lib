import type { LossPoint } from "@ml-vis/core/transformer";

export interface LossSparklineProps {
  history: LossPoint[];
  width?: number;
  height?: number;
}

/** Tiny SVG loss curve — normalised to the observed min/max window. */
export function LossSparkline({ history, width = 150, height = 40 }: LossSparklineProps) {
  const points = history.length;
  if (points < 2) {
    return (
      <svg className="tf-sparkline" width={width} height={height} role="img">
        <line x1={0} y1={height - 2} x2={width} y2={height - 2} stroke="#cbd5e1" strokeDasharray="3 3" />
      </svg>
    );
  }
  const values = history.map((p) => p.loss);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  const coords = history.map((p, i) => {
    const x = (i / (points - 1)) * (width - 4) + 2;
    const y = height - 3 - ((p.loss - min) / span) * (height - 6);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <svg className="tf-sparkline" width={width} height={height} role="img">
      <polyline points={coords.join(" ")} fill="none" stroke="#2563eb" strokeWidth={1.5} />
    </svg>
  );
}
