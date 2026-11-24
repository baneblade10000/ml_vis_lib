import { useEffect, useRef, useState } from "react";
import { rgbToCss, scaleInk, valueToRgb } from "@ml-vis/core/network";

export interface AttentionHeatmapProps {
  title: string;
  hint: string;
  /** Attention probabilities `[head][row][col]` for one layer. */
  matrices: number[][][];
  head: number | "mean";
  rowLabels: string[];
  colLabels: string[];
  /** `cross` styles this heatmap as the cross-attention one. */
  kind?: "self" | "cross";
}

/** Label gutter under the grid while the column labels fit horizontally. */
const LABEL_H = 18;
const MAX_CELL = 34;
/** Cells grow past `MAX_CELL` only as far as needed to fit their column label. */
const MAX_LABEL_CELL = 48;
const LABEL_GAP = 4;
const LABEL_PX = 10;
/** Column labels tilt once the grid is too narrow to fit them upright. */
const LABEL_TILT = -Math.PI / 4;

/**
 * Shared design tokens (`--nn-*` of network-playground.css), as the canvas
 * inherits them. Canvas has no `var()`, so the layout's chrome colors and font
 * are resolved here; cell fills come from the numeric scale instead.
 */
interface CanvasTokens {
  font: string;
  fontBold: string;
  text: string;
  textSecondary: string;
  accent: string;
  surface: string;
  border: string;
}

function readTokens(el: Element): CanvasTokens {
  const styles = getComputedStyle(el);
  const token = (name: string) => styles.getPropertyValue(name).trim();
  const family = token("--nn-font");
  return {
    font: `${LABEL_PX}px ${family}`,
    fontBold: `bold ${LABEL_PX}px ${family}`,
    text: token("--nn-text"),
    textSecondary: token("--nn-text-secondary"),
    accent: token("--nn-accent"),
    surface: token("--nn-surface"),
    border: token("--nn-border-strong"),
  };
}

/** Selects (or averages) one head's probability matrix. */
function pickMatrix(matrices: number[][][], head: number | "mean"): number[][] {
  if (!matrices.length) return [];
  if (head === "mean") {
    return matrices[0].map((row, i) =>
      row.map((_, j) => matrices.reduce((s, m) => s + (m[i]?.[j] ?? 0), 0) / matrices.length),
    );
  }
  return matrices[head] ?? [];
}

interface GridLayout {
  /** Width of the row-label gutter. */
  labelW: number;
  cell: number;
  /** Height of the column-label gutter: labels tilt when a cell cannot fit them. */
  labelH: number;
  tilted: boolean;
}

/**
 * Grid geometry for `width`. Word labels vary in width — the row gutter fits
 * the longest one, and a cell is never narrower than its column label while the
 * grid still fits `width`. Past that the column labels tilt, so neighbouring
 * labels never overlap. `ctx` must have the label font set.
 */
function gridLayout(
  ctx: CanvasRenderingContext2D,
  width: number,
  cols: number,
  rowLabels: string[],
  colLabels: string[],
): GridLayout {
  const longest = Math.max(
    ...rowLabels.map((l) => l.length),
    ...colLabels.map((l) => l.length),
    2,
  );
  // Whole pixels only, or every label and cell edge lands on a half-pixel.
  const labelW = Math.round(Math.min(Math.max(30, longest * 6.4 + 8), 120));
  const roomy = Math.max(Math.floor((width - labelW) / cols), 9);
  const fitted = Math.min(roomy, MAX_CELL);
  const widest = Math.floor(
    Math.max(0, ...colLabels.slice(0, cols).map((l) => ctx.measureText(l).width)),
  );
  const wanted = widest + LABEL_GAP * 2;
  const cell = Math.min(Math.max(fitted, wanted), roomy, MAX_LABEL_CELL);
  const tilted = cell < wanted;
  const labelH = tilted
    ? Math.ceil(widest * Math.SQRT1_2 + LABEL_PX * Math.SQRT1_2 + LABEL_GAP * 2)
    : LABEL_H;
  return { labelW, cell, labelH, tilted };
}

export function AttentionHeatmap({
  title,
  hint,
  matrices,
  head,
  rowLabels,
  colLabels,
  kind = "self",
}: AttentionHeatmapProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null);

  const matrix = pickMatrix(matrices, head);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const rows = matrix.length;
    const cols = matrix[0]?.length ?? 0;
    if (!rows || !cols) return;

    const tokens = readTokens(canvas);

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.font = tokens.font;
      const { labelW, cell, labelH, tilted } = gridLayout(
        ctx,
        wrap.clientWidth,
        cols,
        rowLabels,
        colLabels,
      );
      const gridW = cell * cols;
      const gridH = cell * rows;
      const height = labelH + gridH + 2;
      canvas.width = Math.round((labelW + gridW) * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${labelW + gridW}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = tokens.font;

      ctx.fillStyle = tokens.textSecondary;
      // Column labels (bottom): upright under their cell, or hanging from its centre.
      if (tilted) {
        ctx.textAlign = "right";
        ctx.textBaseline = "top";
        for (let j = 0; j < cols; j++) {
          ctx.save();
          ctx.translate(labelW + j * cell + cell / 2 + LABEL_GAP, gridH + LABEL_GAP);
          ctx.rotate(LABEL_TILT);
          ctx.fillText(colLabels[j] ?? "", 0, 0);
          ctx.restore();
        }
      } else {
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        for (let j = 0; j < cols; j++) {
          ctx.fillText(colLabels[j] ?? "", labelW + j * cell + cell / 2, gridH + LABEL_GAP);
        }
      }
      // Row labels (left); the hovered query gets the ink color.
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i < rows; i++) {
        const hovered = hover?.row === i;
        ctx.font = hovered ? tokens.fontBold : tokens.font;
        ctx.fillStyle = hovered ? tokens.text : tokens.textSecondary;
        ctx.fillText(rowLabels[i] ?? "", labelW - 6, i * cell + cell / 2);
      }
      // Cells.
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const v = matrix[i]?.[j] ?? 0;
          const x = labelW + j * cell;
          const y = i * cell;
          // Attention weights are non-negative: white at 0 → positive token at 1.
          // The floor keeps zero cells faintly outlined like the rest of the grid.
          ctx.fillStyle = rgbToCss(valueToRgb(Math.max(v, 0.03), 1));
          ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
          if (cell >= 24 && v > 0.45) {
            ctx.fillStyle = scaleInk(v, 1);
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(v.toFixed(1).replace(/^0/, ""), x + cell / 2, y + cell / 2);
          }
        }
      }
      // Hover highlight + exact value.
      if (hover && hover.row < rows && hover.col < cols) {
        const x = labelW + hover.col * cell;
        const y = hover.row * cell;
        ctx.strokeStyle = tokens.accent;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x + 0.75, y + 0.75, cell - 1.5, cell - 1.5);
        const v = matrix[hover.row][hover.col];
        const text = v.toFixed(3);
        ctx.font = tokens.fontBold;
        const w = Math.ceil(ctx.measureText(text).width + 10);
        const tx = Math.round(Math.min(Math.max(x + cell / 2 - w / 2, 0), labelW + gridW - w));
        const ty = y - 16 < 0 ? y + cell + 2 : y - 16;
        ctx.fillStyle = tokens.surface;
        ctx.fillRect(tx, ty, w, 14);
        ctx.strokeStyle = tokens.border;
        ctx.lineWidth = 1;
        ctx.strokeRect(tx + 0.5, ty + 0.5, w - 1, 13);
        ctx.fillStyle = tokens.text;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, tx + w / 2, ty + 7);
      }
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(wrap);
    return () => observer.disconnect();
  }, [matrix, rowLabels, colLabels, hover]);

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const cols = matrix[0]?.length ?? 0;
    const ctx = canvas.getContext("2d");
    if (!cols || !ctx) return;
    ctx.font = readTokens(canvas).font;
    const { labelW, cell } = gridLayout(ctx, canvas.clientWidth, cols, rowLabels, colLabels);
    const rect = canvas.getBoundingClientRect();
    const col = Math.floor((e.clientX - rect.left - labelW) / cell);
    const row = Math.floor((e.clientY - rect.top) / cell);
    if (row >= 0 && row < matrix.length && col >= 0 && col < cols) {
      setHover((prev) => (prev?.row === row && prev?.col === col ? prev : { row, col }));
    } else {
      setHover(null);
    }
  };

  return (
    <figure className={`tf-heatmap${kind === "cross" ? " tf-heatmap--cross" : ""}`}>
      <figcaption className="tf-heatmap__title">{title}</figcaption>
      <div ref={wrapRef} className="tf-heatmap__wrap">
        <canvas ref={canvasRef} onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
      </div>
      <p className="tf-heatmap__hint">{hint}</p>
    </figure>
  );
}
