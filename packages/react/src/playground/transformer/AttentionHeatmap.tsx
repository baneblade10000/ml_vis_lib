import { useEffect, useRef, useState } from "react";
import { tokenColor } from "./tokenColors";

export interface AttentionHeatmapProps {
  title: string;
  hint: string;
  /** Attention probabilities `[head][row][col]` for one layer. */
  matrices: number[][][];
  head: number | "mean";
  rowLabels: string[];
  colLabels: string[];
  /** Token ids aligned with labels — colors the axes to match the strips. */
  rowTokens?: number[];
  colTokens?: number[];
  alphabetSize?: number;
}

const LABEL_H = 18;
const MAX_CELL = 34;
const FONT = "10px Inter, system-ui, sans-serif";

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

export function AttentionHeatmap({
  title,
  hint,
  matrices,
  head,
  rowLabels,
  colLabels,
  rowTokens,
  colTokens,
  alphabetSize = 20,
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

    // Word labels vary in width — size the label gutters to the longest one.
    const longest = Math.max(
      ...rowLabels.map((l) => l.length),
      ...colLabels.map((l) => l.length),
      2,
    );
    const labelW = Math.min(Math.max(30, longest * 6.4 + 8), 120);

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = wrap.clientWidth;
      const cell = Math.min(Math.max(Math.floor((width - labelW) / cols), 9), MAX_CELL);
      const gridW = cell * cols;
      const gridH = cell * rows;
      const height = LABEL_H + gridH + 2;
      canvas.width = Math.round((labelW + gridW) * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${labelW + gridW}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ctx.font = FONT;
      // Column labels (bottom), colored by token.
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let j = 0; j < cols; j++) {
        const x = labelW + j * cell + cell / 2;
        ctx.fillStyle = colTokens ? tokenColor(colTokens[j], alphabetSize) : "#64748b";
        ctx.fillText(colLabels[j] ?? "", x, gridH + 4);
      }
      // Row labels (left), colored by token; hovered query gets bolder.
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i < rows; i++) {
        const y = i * cell + cell / 2;
        if (hover?.row === i) {
          ctx.font = "bold 10px Inter, system-ui, sans-serif";
        }
        ctx.fillStyle = rowTokens ? tokenColor(rowTokens[i], alphabetSize) : "#64748b";
        ctx.fillText(rowLabels[i] ?? "", labelW - 6, y);
        if (hover?.row === i) {
          ctx.font = FONT;
        }
      }
      // Cells.
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const v = matrix[i]?.[j] ?? 0;
          const x = labelW + j * cell;
          const y = i * cell;
          ctx.fillStyle = `rgba(37, 99, 235, ${Math.max(v, 0.03)})`;
          ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
          if (cell >= 24 && v > 0.45) {
            ctx.fillStyle = v > 0.7 ? "#ffffff" : "rgba(255,255,255,0.85)";
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
        ctx.strokeStyle = "#1d4ed8";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x + 0.75, y + 0.75, cell - 1.5, cell - 1.5);
        const v = matrix[hover.row][hover.col];
        const text = v.toFixed(3);
        ctx.font = "bold 10px Inter, system-ui, sans-serif";
        const w = ctx.measureText(text).width + 10;
        const tx = Math.min(Math.max(x + cell / 2 - w / 2, 0), labelW + gridW - w);
        const ty = y - 16 < 0 ? y + cell + 2 : y - 16;
        ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
        ctx.fillRect(tx, ty, w, 14);
        ctx.fillStyle = "#f8fafc";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, tx + w / 2, ty + 7);
      }
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(wrap);
    return () => observer.disconnect();
  }, [matrix, rowLabels, colLabels, rowTokens, colTokens, alphabetSize, hover]);

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const cols = matrix[0]?.length ?? 0;
    if (!cols) return;
    const longest = Math.max(
      ...rowLabels.map((l) => l.length),
      ...colLabels.map((l) => l.length),
      2,
    );
    const labelW = Math.min(Math.max(30, longest * 6.4 + 8), 120);
    const cell = Math.min(Math.max(Math.floor((canvas.clientWidth - labelW) / cols), 9), MAX_CELL);
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
    <figure className="tf-heatmap">
      <figcaption className="tf-heatmap__title">{title}</figcaption>
      <div ref={wrapRef} className="tf-heatmap__wrap">
        <canvas ref={canvasRef} onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
      </div>
      <p className="tf-heatmap__hint">{hint}</p>
    </figure>
  );
}
