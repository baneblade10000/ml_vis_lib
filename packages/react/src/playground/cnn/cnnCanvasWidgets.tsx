import { useCallback, useLayoutEffect, useRef } from "react";
import { renderValueMatrix } from "@ml-vis/core/charts";
import { weightColor, weightValueNormalized } from "@ml-vis/core/network";
import { CNN_CELL_PX, cnnGridPx } from "./cnnAdapter";
import type { RfRect } from "./receptiveField";

export function drawRfOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  cell: number,
  rect: RfRect | null,
  kind: "field" | "pixel",
) {
  ctx.clearRect(0, 0, width, height);
  if (!rect) return;
  const x = rect.x0 * cell;
  const y = rect.y0 * cell;
  const w = (rect.x1 - rect.x0 + 1) * cell;
  const h = (rect.y1 - rect.y0 + 1) * cell;
  if (kind === "field") {
    ctx.fillStyle = "rgba(94, 201, 255, 0.32)";
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "#5ec9ff";
    ctx.lineWidth = Math.max(2, cell * 0.35);
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  } else {
    ctx.strokeStyle = "#5ec9ff";
    ctx.lineWidth = Math.max(2, cell * 0.4);
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    ctx.fillStyle = "rgba(52, 87, 163, 0.2)";
    ctx.fillRect(x, y, w, h);
  }
}

/** Map kernel cells onto the same tanh-normalized scale as NN edge weights. */
export function normalizeWeightMap(map: number[][]): number[][] {
  return map.map((row) => row.map((w) => weightValueNormalized(w)));
}

/** Per-filter bias chip — same diverging palette as NN neuron biases. */
export function ConvBiasIndicator({ bias }: { bias: number }) {
  return (
    <span
      className="cnn-filter-bias"
      data-sign={bias >= 0 ? "pos" : "neg"}
      aria-hidden
      title={`bias ${bias.toFixed(3)}`}
      style={{ background: weightColor(weightValueNormalized(bias)) }}
    />
  );
}

/**
 * Mini pixelated kernel — same {@link CNN_CELL_PX} as feature maps.
 */
export function KernelMini({
  map,
  label,
  bias,
  selected = false,
  interactive = false,
  onSelect,
}: {
  map: number[][];
  label?: string;
  bias?: number;
  selected?: boolean;
  interactive?: boolean;
  onSelect?: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatRef = useRef<HTMLCanvasElement>(null);
  const rows = map.length;
  const cols = map[0]?.length ?? 0;
  const { w, h } = cnnGridPx(Math.max(1, rows), Math.max(1, cols));

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const heat = heatRef.current;
    if (!canvas || !heat || !rows || !cols) return;
    renderValueMatrix(heat, normalizeWeightMap(map), {
      layout: "row-major",
      palette: "diverging",
    });
    const { w: bw, h: bh } = cnnGridPx(rows, cols);
    canvas.width = bw;
    canvas.height = bh;
    canvas.style.width = `${bw}px`;
    canvas.style.height = `${bh}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, bw, bh);
    ctx.drawImage(heat, 0, 0, bw, bh);
  }, [map, rows, cols]);

  const paintRef = useRef(paint);
  paintRef.current = paint;
  useLayoutEffect(() => paintRef.current(), [paint]);

  if (!rows || !cols) {
    return (
      <div
        className="cnn-kernel-mini cnn-kernel-mini--empty"
        style={{ width: w, height: h }}
        title={label}
      />
    );
  }

  const title =
    typeof bias === "number" && Number.isFinite(bias)
      ? `${label ?? "kernel"} · bias ${bias.toFixed(3)}`
      : label;

  return (
    <div
      className={`cnn-kernel-with-bias${interactive ? " cnn-kernel-with-bias--pick nodrag nopan" : ""}${selected ? " cnn-kernel-with-bias--selected" : ""}`}
      title={title}
      onPointerDown={
        interactive && onSelect
          ? (e) => {
              e.stopPropagation();
              onSelect();
            }
          : undefined
      }
    >
      {typeof bias === "number" && Number.isFinite(bias) && (
        <ConvBiasIndicator bias={bias} />
      )}
      <div className="cnn-kernel-mini" style={{ width: w, height: h }}>
        <canvas ref={heatRef} width={cols} height={rows} hidden aria-hidden />
        <canvas ref={canvasRef} className="cnn-feature-canvas" width={w} height={h} />
      </div>
    </div>
  );
}

/** Render a 1-D signal — each sample is {@link CNN_CELL_PX} wide. */
export function Signal1DCanvas({ values }: { values: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatRef = useRef<HTMLCanvasElement>(null);
  const n = Math.max(1, values.length);
  const bw = n * CNN_CELL_PX;
  const bh = CNN_CELL_PX;

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const heat = heatRef.current;
    if (!canvas || !heat || !values.length) return;
    const mat = [values.slice()];
    renderValueMatrix(heat, mat, { layout: "row-major", palette: "gray" });
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
      canvas.style.width = `${bw}px`;
      canvas.style.height = `${bh}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, bw, bh);
    ctx.drawImage(heat, 0, 0, bw, bh);
  }, [values, bw, bh]);

  const paintRef = useRef(paint);
  paintRef.current = paint;
  useLayoutEffect(() => paintRef.current(), [paint]);

  return (
    <div className="cnn-feature-cell cnn-feature-cell--signal" style={{ width: bw, height: bh }}>
      <canvas ref={heatRef} width={1} height={1} hidden aria-hidden />
      <canvas ref={canvasRef} className="cnn-feature-canvas" />
    </div>
  );
}

/** Activation → grayscale fill for flatten unit squares. */
export function activationGray(values: number[]): string[] {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || max - min < 1e-9) {
    min -= 0.5;
    max += 0.5;
  }
  const span = max - min;
  return values.map((v) => {
    const t = Math.min(1, Math.max(0, (v - min) / span));
    const g8 = Math.round(Math.pow(t, 0.85) * 255);
    return `rgb(${g8},${g8},${g8})`;
  });
}

/** Flatten / GAP column values from feature-map `signals`. */
export function flatSignalValues(signals: number[][] | undefined, length?: number): number[] {
  if (!signals?.length) return new Array(length || 0).fill(0);
  // Dense / flatten / fixed GAP export: [[v0..vN]]
  if (signals.length === 1) {
    const row = signals[0]!;
    if (row.length) return row;
  }
  // Legacy GAP [C][1] (or multi-row): one scalar per channel/row.
  const fromRows = signals.map((row) => row[0] ?? 0);
  if (fromRows.some((v) => v !== 0) || !length) return fromRows;
  return new Array(length).fill(0);
}
