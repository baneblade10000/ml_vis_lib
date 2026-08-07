import { CLASS_0_HEX, CLASS_1_HEX, valueToRgb } from "../playground/network/colors";

export type CurvePoint = { x: number; y: number; label: number };

/**
 * Draw a 1D activation/prediction curve into a canvas (values typically in [-1, 1]
 * or a wider regression range; y is auto-scaled with padding).
 */
export function renderCurve(
  canvas: HTMLCanvasElement,
  values: number[],
  options?: {
    xDomain?: [number, number];
    yDomain?: [number, number];
    stroke?: string;
    fill?: boolean;
    lineWidth?: number;
    /** Clear canvas before drawing (default true). */
    clear?: boolean;
    /** Draw the y=0 baseline (default true). */
    baseline?: boolean;
  },
): void {
  const n = values.length;
  if (n < 2) return;

  const width = canvas.width;
  const height = canvas.height;
  if (!width || !height) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const yDomain = options?.yDomain ?? inferYDomain(values);
  const [yMin, yMax] = yDomain;
  const stroke = options?.stroke ?? "rgb(90, 90, 110)";
  // Keep strokes thin relative to CSS size (canvas may be DPR-scaled).
  const lineWidth = options?.lineWidth ?? Math.max(1.25, width / 160);

  if (options?.clear !== false) {
    ctx.clearRect(0, 0, width, height);
  }
  ctx.imageSmoothingEnabled = true;

  // Zero baseline
  const zeroY = mapY(0, yMin, yMax, height);
  if (options?.baseline !== false) {
    ctx.strokeStyle = "rgba(0, 0, 0, 0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(width, zeroY);
    ctx.stroke();
  }

  const pts = new Array<{ x: number; y: number }>(n);
  for (let i = 0; i < n; i++) {
    pts[i] = {
      x: (i / (n - 1)) * width,
      y: mapY(values[i]!, yMin, yMax, height),
    };
  }

  const strokePath = () => {
    ctx.beginPath();
    ctx.moveTo(pts[0]!.x, pts[0]!.y);
    if (n === 2) {
      ctx.lineTo(pts[1]!.x, pts[1]!.y);
      return;
    }
    // Midpoint chain: smooth polyline without overshoot (no Catmull spikes).
    for (let i = 1; i < n - 1; i++) {
      const midX = (pts[i]!.x + pts[i + 1]!.x) / 2;
      const midY = (pts[i]!.y + pts[i + 1]!.y) / 2;
      ctx.quadraticCurveTo(pts[i]!.x, pts[i]!.y, midX, midY);
    }
    const last = pts[n - 1]!;
    const prev = pts[n - 2]!;
    ctx.quadraticCurveTo(prev.x, prev.y, last.x, last.y);
  };

  if (options?.fill !== false) {
    strokePath();
    ctx.lineTo(width, zeroY);
    ctx.lineTo(0, zeroY);
    ctx.closePath();
    ctx.fillStyle = "rgba(120, 120, 160, 0.1)";
    ctx.fill();
  }

  strokePath();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();
}

/** Draw train points on a curve canvas (call before the curve to keep them underneath). */
export function renderCurvePoints(
  canvas: HTMLCanvasElement,
  points: CurvePoint[],
  options?: {
    xDomain?: [number, number];
    yDomain?: [number, number];
    colorByLabel?: boolean;
    pointColor?: string;
  },
): void {
  const width = canvas.width;
  const height = canvas.height;
  if (!width || !height || !points.length) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const xDomain = options?.xDomain ?? [-6, 6];
  const yDomain = options?.yDomain ?? [-1.2, 1.2];
  const [xMin, xMax] = xDomain;
  const [yMin, yMax] = yDomain;
  const colorByLabel = options?.colorByLabel ?? true;
  const radius = Math.max(1.35, width / 110);

  for (const p of points) {
    const px = ((p.x - xMin) / (xMax - xMin)) * width;
    const py = mapY(p.y, yMin, yMax, height);
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    if (colorByLabel) {
      ctx.fillStyle = p.label > 0 ? CLASS_1_HEX : CLASS_0_HEX;
      ctx.globalAlpha = 0.75;
    } else {
      ctx.fillStyle = options?.pointColor ?? "rgb(100, 116, 139)";
      ctx.globalAlpha = 0.55;
    }
    ctx.fill();
    ctx.globalAlpha = 1;
  }
}

/** Draw a dashed target curve (regression ground truth). */
export function renderTargetCurve(
  canvas: HTMLCanvasElement,
  values: number[],
  options?: {
    yDomain?: [number, number];
    stroke?: string;
  },
): void {
  const n = values.length;
  if (n < 2) return;
  const width = canvas.width;
  const height = canvas.height;
  if (!width || !height) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const yDomain = options?.yDomain ?? inferYDomain(values);
  const [yMin, yMax] = yDomain;

  const pts = new Array<{ x: number; y: number }>(n);
  for (let i = 0; i < n; i++) {
    pts[i] = {
      x: (i / (n - 1)) * width,
      y: mapY(values[i]!, yMin, yMax, height),
    };
  }
  ctx.beginPath();
  ctx.moveTo(pts[0]!.x, pts[0]!.y);
  if (n === 2) {
    ctx.lineTo(pts[1]!.x, pts[1]!.y);
  } else {
    for (let i = 1; i < n - 1; i++) {
      const midX = (pts[i]!.x + pts[i + 1]!.x) / 2;
      const midY = (pts[i]!.y + pts[i + 1]!.y) / 2;
      ctx.quadraticCurveTo(pts[i]!.x, pts[i]!.y, midX, midY);
    }
    const last = pts[n - 1]!;
    const prev = pts[n - 2]!;
    ctx.quadraticCurveTo(prev.x, prev.y, last.x, last.y);
  }
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = options?.stroke ?? "rgba(40, 40, 50, 0.45)";
  ctx.lineWidth = Math.max(1.1, width / 180);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();
  ctx.setLineDash([]);
}

/** Color a curve stroke from the mean activation using the heatmap palette. */
export function curveStrokeFromValues(values: number[]): string {
  if (!values.length) return "rgb(90, 90, 110)";
  let sum = 0;
  for (const v of values) sum += v;
  const { r, g, b } = valueToRgb(sum / values.length);
  return `rgb(${r}, ${g}, ${b})`;
}

export function inferYDomain(values: number[], pad = 0.15): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [-1.2, 1.2];
  if (min === max) {
    min -= 1;
    max += 1;
  }
  // Keep zero in view when values are small.
  min = Math.min(min, -0.2);
  max = Math.max(max, 0.2);
  const span = max - min;
  return [min - span * pad, max + span * pad];
}

function mapY(value: number, yMin: number, yMax: number, height: number): number {
  return (1 - (value - yMin) / (yMax - yMin)) * height;
}
