import { reduceMatrix, renderValueMatrix } from "@ml-vis/core/charts";
import {
  CLASS_0_HEX,
  CLASS_1_HEX,
  PLAY_DISPLAY_DENSITY,
  valueToRgb,
  weightColor,
  weightValueNormalized,
} from "@ml-vis/core/network";
import type { DataPoint } from "./graphAdapter";

export function matrixForDisplay(
  matrix: number[][],
  displayPx: number,
  coarseTo?: number,
  live = false,
): number[][] {
  if (live) {
    let factor = Math.max(1, Math.floor(matrix.length / PLAY_DISPLAY_DENSITY));
    while (factor > 1 && matrix.length % factor !== 0) factor -= 1;
    return factor > 1 ? reduceMatrix(matrix, factor) : matrix;
  }
  if (coarseTo !== undefined && matrix.length <= coarseTo) {
    return matrix;
  }
  let grid = matrix;
  if (coarseTo !== undefined && grid.length > coarseTo) {
    let factor = Math.max(1, Math.round(grid.length / coarseTo));
    while (factor > 1 && grid.length % factor !== 0) factor -= 1;
    if (factor > 1) grid = reduceMatrix(grid, factor);
  }
  if (coarseTo !== undefined) {
    return grid;
  }
  const dpr = window.devicePixelRatio || 1;
  const target = Math.ceil(displayPx * dpr);
  let factor = Math.max(1, Math.floor(grid.length / target));
  while (factor > 1 && grid.length % factor !== 0) factor -= 1;
  return factor > 1 ? reduceMatrix(grid, factor) : grid;
}

const OUTPUT_X_DOMAIN: [number, number] = [-6, 6];

export function ensureCanvasSize(
  canvas: HTMLCanvasElement,
  px: number,
  dpr: number,
): CanvasRenderingContext2D | null {
  const w = Math.round(px * dpr);
  const h = w;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${px}px`;
    canvas.style.height = `${px}px`;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

export function paintTrainOverlay(
  canvas: HTMLCanvasElement,
  px: number,
  trainData: DataPoint[],
  regression = false,
): void {
  const dpr = window.devicePixelRatio || 1;
  const ctx = ensureCanvasSize(canvas, px, dpr);
  if (!ctx) return;
  ctx.clearRect(0, 0, px, px);
  const [minX, maxX] = OUTPUT_X_DOMAIN;
  const mapX = (x: number) => ((x - minX) / (maxX - minX)) * px;
  const mapY = (y: number) => (1 - (y - minX) / (maxX - minX)) * px;
  for (const point of trainData) {
    ctx.beginPath();
    ctx.arc(mapX(point.x), mapY(point.y), 2.5, 0, Math.PI * 2);
    if (regression) {
      const { r, g, b } = valueToRgb(point.label);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    } else {
      ctx.fillStyle = point.label > 0 ? CLASS_1_HEX : CLASS_0_HEX;
    }
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
    ctx.lineWidth = 0.75;
    ctx.stroke();
  }
}

export function paintHeatmapCanvas(
  canvas: HTMLCanvasElement,
  heatmap: HTMLCanvasElement,
  matrix: number[][],
  size: number,
  discretize: boolean,
  smooth: boolean,
  coarseTo?: number,
  live = false,
): void {
  const px = size - 6;
  const mini = coarseTo !== undefined;
  const reduced = matrixForDisplay(matrix, px, coarseTo, live);
  renderValueMatrix(heatmap, reduced, discretize);

  const dpr = mini ? 1 : window.devicePixelRatio || 1;
  const ctx = ensureCanvasSize(canvas, px, dpr);
  if (!ctx) return;
  ctx.imageSmoothingEnabled = smooth;
  ctx.imageSmoothingQuality = smooth ? "high" : "low";
  ctx.drawImage(heatmap, 0, 0, px, px);
}

export function BiasIndicator({ bias }: { bias: number }) {
  // The bias square sits in the neuron's bottom-right corner. Its fill color
  // encodes the bias via the diverging palette: deep blue for negative, sky cyan
  // for positive, with the hue saturating toward the palette extremes as |bias|
  // grows (tanh-normalized, so small biases stay readable).
  return (
    <span
      className="nn-flow-bias"
      data-sign={bias >= 0 ? "pos" : "neg"}
      aria-hidden
      style={{ background: weightColor(weightValueNormalized(bias)) }}
    />
  );
}
