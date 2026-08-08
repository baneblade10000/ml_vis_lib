/**
 * Generic 1-D signal plot used by every tab of the Signal lab.
 *
 * One plot draws an arbitrary number of series (curves or stem bars), a shared
 * x-axis, per-series y-domains, an optional zero baseline, a legend, and
 * vertical "marker" lines (e.g. the sliding kernel position in the convolution
 * tab). Built to the same lifecycle the rest of the codebase expects
 * (`setData`/`setSize`/`resize`/`destroy`, DPR-scaled via `setupCanvas`).
 */

import { roundRect, setupCanvas } from "./common";

export type SignalPlotStyle = "curve" | "stem";

export interface SignalPlotSeries {
  values: ArrayLike<number>;
  color: string;
  /** Localised label is supplied by the React layer. */
  label?: string;
  style?: SignalPlotStyle;
  dashed?: boolean;
  /** Fill under a curve series (default false). */
  fill?: boolean;
  /** Hide this series from the legend (default false). */
  hiddenInLegend?: boolean;
}

export interface SignalMarker {
  /** x position in sample units (0-based index into the x-domain). */
  x: number;
  color: string;
  label?: string;
  dashed?: boolean;
}

export interface SignalPlotPayload {
  series: SignalPlotSeries[];
  markers?: SignalMarker[];
  /** Optional caption drawn in the top-right corner. */
  caption?: string;
  /** Optional formula/recipe drawn as a faint subtitle. */
  subtitle?: string;
  /** Hide the legend (default false). */
  hideLegend?: boolean;
}

interface PlotLayout {
  pad: number;
  plotW: number;
  plotH: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  mapX: (x: number) => number;
  mapY: (y: number) => number;
}

export class SignalPlot {
  private canvas: HTMLCanvasElement;
  private width = 640;
  private height = 360;
  private payload: SignalPlotPayload = { series: [] };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  setData(payload: SignalPlotPayload): void {
    this.payload = payload;
    this.draw();
  }

  setSize(width: number, height: number): void {
    this.width = Math.max(240, width);
    this.height = Math.max(180, height);
    this.draw();
  }

  resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(240, rect.width || this.canvas.clientWidth);
    this.height = Math.max(180, rect.height || this.canvas.clientHeight);
    this.draw();
  }

  destroy(): void {
    const ctx = this.canvas.getContext("2d");
    ctx?.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  private draw(): void {
    const ctx = setupCanvas(this.canvas, this.width, this.height);
    ctx.clearRect(0, 0, this.width, this.height);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, this.width, this.height);

    const layout = this.resolveLayout();
    this.drawGrid(ctx, layout);
    this.drawMarkers(ctx, layout);
    for (const series of this.payload.series) {
      if (series.style === "stem") this.drawStem(ctx, series, layout);
      else this.drawCurve(ctx, series, layout);
    }
    this.drawAxes(ctx, layout);
    if (this.payload.subtitle) this.drawSubtitle(ctx);
    if (!this.payload.hideLegend) this.drawLegend(ctx);
    if (this.payload.caption) this.drawCaption(ctx);
  }

  private resolveLayout(): PlotLayout {
    const pad = 32;
    const plotW = Math.max(1, this.width - pad * 2);
    const plotH = Math.max(1, this.height - pad * 2);

    let xMax = 0;
    let yMin = 0;
    let yMax = 0;
    for (const series of this.payload.series) {
      xMax = Math.max(xMax, series.values.length - 1);
      for (let i = 0; i < series.values.length; i++) {
        const v = series.values[i];
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
    }
    for (const marker of this.payload.markers ?? []) {
      xMax = Math.max(xMax, marker.x);
    }
    if (xMax < 1) xMax = 1;

    // Pad y-domain symmetrically around zero so the baseline stays centred.
    const yAbs = Math.max(Math.abs(yMin), Math.abs(yMax), 0.5);
    yMin = -yAbs * 1.12;
    yMax = yAbs * 1.12;

    const xMin = 0;
    const spanX = Math.max(xMax - xMin, 1e-9);
    const spanY = Math.max(yMax - yMin, 1e-9);
    const mapX = (x: number) => pad + ((x - xMin) / spanX) * plotW;
    const mapY = (y: number) => pad + (1 - (y - yMin) / spanY) * plotH;

    return { pad, plotW, plotH, xMin, xMax, yMin, yMax, mapX, mapY };
  }

  private drawGrid(ctx: CanvasRenderingContext2D, layout: PlotLayout): void {
    const { pad, plotW, plotH, mapY } = layout;
    ctx.save();
    ctx.strokeStyle = "rgba(15, 23, 42, 0.06)";
    ctx.lineWidth = 1;
    // Four horizontal grid lines.
    for (let i = 1; i <= 3; i++) {
      const y = pad + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(pad + plotW, y);
      ctx.stroke();
    }
    // Zero baseline slightly stronger.
    ctx.strokeStyle = "rgba(15, 23, 42, 0.18)";
    ctx.beginPath();
    ctx.moveTo(pad, mapY(0));
    ctx.lineTo(pad + plotW, mapY(0));
    ctx.stroke();
    ctx.restore();
  }

  private drawAxes(ctx: CanvasRenderingContext2D, layout: PlotLayout): void {
    const { pad, plotW, plotH } = layout;
    ctx.save();
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, plotW, plotH);
    ctx.restore();
  }

  private drawCurve(ctx: CanvasRenderingContext2D, series: SignalPlotSeries, layout: PlotLayout): void {
    const values = series.values;
    const n = values.length;
    if (n === 0) return;
    const { mapX, mapY } = layout;

    const pts = new Array<{ x: number; y: number }>(n);
    for (let i = 0; i < n; i++) pts[i] = { x: mapX(i), y: mapY(values[i]) };

    const path = () => {
      ctx.beginPath();
      ctx.moveTo(pts[0]!.x, pts[0]!.y);
      if (n === 2) {
        ctx.lineTo(pts[1]!.x, pts[1]!.y);
        return;
      }
      for (let i = 1; i < n - 1; i++) {
        const midX = (pts[i]!.x + pts[i + 1]!.x) / 2;
        const midY = (pts[i]!.y + pts[i + 1]!.y) / 2;
        ctx.quadraticCurveTo(pts[i]!.x, pts[i]!.y, midX, midY);
      }
      ctx.quadraticCurveTo(pts[n - 2]!.x, pts[n - 2]!.y, pts[n - 1]!.x, pts[n - 1]!.y);
    };

    if (series.fill) {
      const zeroY = mapY(0);
      path();
      ctx.lineTo(pts[n - 1]!.x, zeroY);
      ctx.lineTo(pts[0]!.x, zeroY);
      ctx.closePath();
      ctx.fillStyle = withAlpha(series.color, 0.12);
      ctx.fill();
    }

    ctx.save();
    if (series.dashed) ctx.setLineDash([6, 4]);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = series.color;
    ctx.lineWidth = Math.max(1.5, this.width / 240);
    path();
    ctx.stroke();
    ctx.restore();
  }

  private drawStem(ctx: CanvasRenderingContext2D, series: SignalPlotSeries, layout: PlotLayout): void {
    const values = series.values;
    const n = values.length;
    if (n === 0) return;
    const { mapX, mapY } = layout;
    const zeroY = mapY(0);
    const barW = Math.max(1, (layout.plotW / Math.max(n - 1, 1)) * 0.5);

    ctx.save();
    ctx.strokeStyle = series.color;
    ctx.fillStyle = series.color;
    ctx.lineWidth = Math.max(1, this.width / 320);
    for (let i = 0; i < n; i++) {
      const v = values[i];
      const x = mapX(i);
      const y = mapY(v);
      ctx.beginPath();
      ctx.moveTo(x, zeroY);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, barW * 0.5), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  private drawMarkers(ctx: CanvasRenderingContext2D, layout: PlotLayout): void {
    const markers = this.payload.markers;
    if (!markers?.length) return;
    const { pad, plotH, mapX } = layout;
    ctx.save();
    for (const marker of markers) {
      const x = mapX(marker.x);
      if (marker.dashed) ctx.setLineDash([4, 4]);
      ctx.strokeStyle = marker.color;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.moveTo(x, pad);
      ctx.lineTo(x, pad + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      if (marker.label) {
        ctx.fillStyle = marker.color;
        ctx.font = "11px Helvetica, Arial, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(marker.label, x, pad - 6);
      }
    }
    ctx.textAlign = "left";
    ctx.restore();
  }

  private drawLegend(ctx: CanvasRenderingContext2D): void {
    const entries = this.payload.series.filter((s) => s.label && !s.hiddenInLegend);
    if (!entries.length) return;
    ctx.save();
    ctx.font = "12px Helvetica, Arial, sans-serif";
    const pad = 32;
    let maxW = 0;
    for (const entry of entries) maxW = Math.max(maxW, ctx.measureText(entry.label!).width);
    const rowH = 16;
    const boxW = maxW + 30;
    const boxH = entries.length * rowH + 10;
    const x = this.width - pad - boxW;
    const y = pad + 6;

    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    ctx.strokeStyle = "rgba(15, 23, 42, 0.1)";
    ctx.lineWidth = 1;
    roundRect(ctx, x, y, boxW, boxH, 6);
    ctx.fill();
    ctx.stroke();

    entries.forEach((entry, i) => {
      const cy = y + 10 + i * rowH + rowH / 2 - 1;
      ctx.save();
      if (entry.dashed) ctx.setLineDash([5, 3]);
      ctx.strokeStyle = entry.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x + 10, cy);
      ctx.lineTo(x + 26, cy);
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = "#475569";
      ctx.fillText(entry.label!, x + 32, cy + 4);
    });
    ctx.restore();
  }

  private drawSubtitle(ctx: CanvasRenderingContext2D): void {
    ctx.save();
    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px Helvetica, Arial, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(this.payload.subtitle ?? "", 32, 22);
    ctx.restore();
  }

  private drawCaption(ctx: CanvasRenderingContext2D): void {
    ctx.save();
    ctx.fillStyle = "#475569";
    ctx.font = "12px Helvetica, Arial, sans-serif";
    ctx.textAlign = "left";
    const y = this.height - 10;
    ctx.fillText(this.payload.caption ?? "", 32, y);
    ctx.restore();
  }
}

/** Convert a `#rrggbb` hex colour to an `rgba()` string with the given alpha. */
function withAlpha(color: string, alpha: number): string {
  if (color.startsWith("#") && color.length === 7) {
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  return color;
}
