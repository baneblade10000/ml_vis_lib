/**
 * Decision boundary replay chart.
 * Heatmap rendering follows TensorFlow Playground (Apache-2.0, Google):
 * ImageData per grid cell + smooth CSS scaling.
 */

import type { GridPoint, HistoryRow, Sample } from "../playground/types";
import { probabilityToRgba, roundRect, setupCanvas } from "./common";
import { CLASS_0_HEX, CLASS_1_HEX } from "../playground/tf/colors";

export interface DecisionBoundaryPayload {
  samples: Sample[];
  grid: GridPoint[];
  previousGrid?: GridPoint[];
  history: HistoryRow[];
  epoch: number;
  frameLabel?: string;
}

export interface PlotPointer {
  clientX: number;
  clientY: number;
}

interface PlotGeometry {
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
  pad: number;
  plotW: number;
  plotH: number;
  mapX: (x: number) => number;
  mapY: (y: number) => number;
  gridSize: number;
}

function inferGridSize(grid: GridPoint[]): number {
  if (!grid.length) return 0;
  const side = Math.round(Math.sqrt(grid.length));
  return side * side === grid.length ? side : Math.ceil(Math.sqrt(grid.length));
}

/** Flat grid (row = x2, col = x1) → matrix[col][row] like TF Playground. */
function gridToMatrix(grid: GridPoint[]): number[][] | null {
  const size = inferGridSize(grid);
  if (!size) return null;

  const xs = [...new Set(grid.map((p) => p.x1))].sort((a, b) => a - b);
  const ys = [...new Set(grid.map((p) => p.x2))].sort((a, b) => a - b);
  if (xs.length !== size || ys.length !== size) return null;

  const matrix: number[][] = Array.from({ length: size }, () => new Array<number>(size).fill(0.5));
  for (const point of grid) {
    const col = xs.findIndex((x) => Math.abs(x - point.x1) < 1e-9);
    const row = ys.findIndex((y) => Math.abs(y - point.x2) < 1e-9);
    if (col >= 0 && row >= 0) matrix[col][row] = point.probability;
  }
  return matrix;
}

export class DecisionBoundaryPlot {
  private canvas: HTMLCanvasElement;
  private width = 640;
  private height = 420;
  private heatmapCanvas: HTMLCanvasElement | null = null;
  private payload: DecisionBoundaryPayload = {
    samples: [],
    grid: [],
    history: [],
    epoch: 0,
  };
  private pointer: { px: number; py: number } | null = null;
  private geometry: PlotGeometry | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  setData(payload: DecisionBoundaryPayload): void {
    this.payload = payload;
    this.draw();
  }

  setSize(width: number, height: number): void {
    this.width = Math.max(320, width);
    this.height = Math.max(280, height);
    this.draw();
  }

  setPointer(pointer: PlotPointer | null): void {
    if (!pointer) {
      this.pointer = null;
      this.draw();
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    this.pointer = {
      px: pointer.clientX - rect.left,
      py: pointer.clientY - rect.top,
    };
    this.draw();
  }

  resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(320, rect.width || this.canvas.clientWidth);
    this.height = Math.max(280, rect.height || this.canvas.clientHeight);
    this.draw();
  }

  destroy(): void {
    const ctx = this.canvas.getContext("2d");
    ctx?.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.heatmapCanvas = null;
  }

  nearestGridPoint(clientX: number, clientY: number): { x1: number; x2: number; probability: number } | null {
    if (!this.geometry) return null;
    const rect = this.canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const { pad, plotW, plotH, bounds } = this.geometry;
    if (px < pad || px > pad + plotW || py < pad || py > pad + plotH) return null;

    const x1 = bounds.minX + ((px - pad) / plotW) * (bounds.maxX - bounds.minX);
    const x2 = bounds.maxY - ((py - pad) / plotH) * (bounds.maxY - bounds.minY);

    let best: GridPoint | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const row of this.payload.grid) {
      const dx = row.x1 - x1;
      const dy = row.x2 - x2;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = row;
      }
    }
    return best ? { x1: best.x1, x2: best.x2, probability: best.probability } : null;
  }

  private draw(): void {
    const ctx = setupCanvas(this.canvas, this.width, this.height);
    ctx.clearRect(0, 0, this.width, this.height);

    const mainHeight = Math.round(this.height * 0.72);
    this.geometry = this.drawBoundary(
      ctx,
      this.payload.grid,
      this.payload.samples,
      this.width,
      mainHeight,
      this.payload.frameLabel ?? `epoch ${this.payload.epoch}`,
    );
    this.drawHistory(
      ctx,
      this.payload.history,
      22,
      mainHeight + 24,
      this.width - 44,
      Math.max(70, this.height - mainHeight - 40),
      this.payload.epoch,
    );
  }

  private resolveBounds(samples: Sample[], grid: GridPoint[]) {
    let minX = -1.35;
    let maxX = 1.35;
    let minY = -1.35;
    let maxY = 1.35;
    for (const sample of samples) {
      minX = Math.min(minX, sample.x1);
      maxX = Math.max(maxX, sample.x1);
      minY = Math.min(minY, sample.x2);
      maxY = Math.max(maxY, sample.x2);
    }
    for (const point of grid) {
      minX = Math.min(minX, point.x1);
      maxX = Math.max(maxX, point.x1);
      minY = Math.min(minY, point.x2);
      maxY = Math.max(maxY, point.x2);
    }
    return { minX, maxX, minY, maxY };
  }

  /** TF Playground style: ImageData at grid resolution, scaled to plot area. */
  private drawHeatmap(
    ctx: CanvasRenderingContext2D,
    grid: GridPoint[],
    x: number,
    y: number,
    width: number,
    height: number,
  ): number {
    const matrix = gridToMatrix(grid);
    const size = matrix?.length ?? 0;
    if (!matrix || !size) return 0;

    if (!this.heatmapCanvas || this.heatmapCanvas.width !== size) {
      this.heatmapCanvas = document.createElement("canvas");
      this.heatmapCanvas.width = size;
      this.heatmapCanvas.height = size;
    }

    const heatCtx = this.heatmapCanvas.getContext("2d");
    if (!heatCtx) return size;

    const image = heatCtx.createImageData(size, size);
    for (let row = 0, p = -1; row < size; row++) {
      for (let col = 0; col < size; col++) {
        const [r, g, b, a] = probabilityToRgba(matrix[col][row], 160);
        image.data[++p] = r;
        image.data[++p] = g;
        image.data[++p] = b;
        image.data[++p] = a;
      }
    }
    heatCtx.putImageData(image, 0, 0);

    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(this.heatmapCanvas, x, y, width, height);
    ctx.restore();
    return size;
  }

  private drawBoundary(
    ctx: CanvasRenderingContext2D,
    grid: GridPoint[],
    samples: Sample[],
    width: number,
    height: number,
    label: string,
  ): PlotGeometry {
    const pad = 24;
    const plotW = width - pad * 2;
    const plotH = height - pad * 2;
    const bounds = this.resolveBounds(samples, grid);
    const mapX = (x: number) => pad + ((x - bounds.minX) / Math.max(bounds.maxX - bounds.minX, 1e-9)) * plotW;
    const mapY = (y: number) => pad + (1 - (y - bounds.minY) / Math.max(bounds.maxY - bounds.minY, 1e-9)) * plotH;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const gridSize = this.drawHeatmap(ctx, grid, pad, pad, plotW, plotH);

    ctx.strokeStyle = "#d0d0d0";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, plotW, plotH);

    ctx.fillStyle = "#555";
    ctx.font = "13px Helvetica, Arial, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Output", pad, 16);
    ctx.textAlign = "right";
    ctx.fillStyle = "#888";
    ctx.fillText(label, width - pad, 16);
    ctx.textAlign = "left";

    for (const row of samples) {
      const target = row.target;
      ctx.beginPath();
      ctx.arc(mapX(row.x1), mapY(row.x2), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = target > 0 ? CLASS_1_HEX : CLASS_0_HEX;
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.15)";
      ctx.lineWidth = 0.75;
      ctx.stroke();
    }

    if (this.pointer) this.drawPointer(ctx, this.pointer, { pad, plotW, plotH, mapX, mapY });
    return { bounds, pad, plotW, plotH, mapX, mapY, gridSize };
  }

  private drawPointer(
    ctx: CanvasRenderingContext2D,
    pointer: { px: number; py: number },
    geometry: Pick<PlotGeometry, "pad" | "plotW" | "plotH" | "mapX" | "mapY">,
  ): void {
    const { px, py } = pointer;
    if (px < geometry.pad || px > geometry.pad + geometry.plotW || py < geometry.pad || py > geometry.pad + geometry.plotH) {
      return;
    }
    ctx.save();
    ctx.strokeStyle = "rgba(0,0,0,0.2)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(geometry.pad, py);
    ctx.lineTo(geometry.pad + geometry.plotW, py);
    ctx.moveTo(px, geometry.pad);
    ctx.lineTo(px, geometry.pad + geometry.plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  private drawHistory(
    ctx: CanvasRenderingContext2D,
    history: HistoryRow[],
    x: number,
    y: number,
    width: number,
    height: number,
    currentEpoch: number,
  ): void {
    ctx.fillStyle = "#ffffff";
    roundRect(ctx, x, y, width, height, 4);
    ctx.fill();
    ctx.strokeStyle = "#e0e0e0";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = "#666";
    ctx.font = "12px Helvetica, Arial, sans-serif";
    ctx.fillText("Validation accuracy", x + 10, y + 16);

    if (!history.length) return;
    const values = history.map((row) => row.validationAccuracy);
    const min = Math.min(...values, 0);
    const max = Math.max(...values, 1);
    ctx.strokeStyle = CLASS_0_HEX;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const points: Array<{ x: number; y: number; epoch: number }> = [];
    values.forEach((value, index) => {
      const px = x + 10 + (index / Math.max(values.length - 1, 1)) * (width - 20);
      const py = y + height - 12 - ((value - min) / Math.max(max - min, 1e-9)) * (height - 32);
      points.push({ x: px, y: py, epoch: history[index]?.epoch ?? index });
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    const marker = points.reduce((best, point) =>
      Math.abs(point.epoch - currentEpoch) < Math.abs(best.epoch - currentEpoch) ? point : best,
    );
    ctx.beginPath();
    ctx.arc(marker.x, marker.y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = CLASS_1_HEX;
    ctx.fill();
  }
}

export type { DecisionBoundaryPayload as DecisionBoundaryPlotPayload };
