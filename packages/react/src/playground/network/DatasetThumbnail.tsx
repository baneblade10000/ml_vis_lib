import { useEffect, useRef } from "react";
import {
  CLASS_0_HEX,
  CLASS_1_HEX,
  DATASETS,
  DATASETS_1D_CLASSIFICATION,
  DATASETS_1D_REGRESSION,
  valueToRgb,
  type NetworkAnyDatasetId,
  type NetworkDataMode,
  type DatasetId,
  type NetworkProblemType,
} from "@ml-vis/core/network";

/** Small canvas preview of a dataset, used by the dataset picker. */
export function DatasetThumbnail({
  datasetId,
  label,
  selected,
  onSelect,
  mode,
  problemType = "classification",
}: {
  datasetId: NetworkAnyDatasetId;
  label: string;
  selected: boolean;
  onSelect: () => void;
  mode: NetworkDataMode;
  problemType?: NetworkProblemType;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const size = 128;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, size, size);

    if (mode === "1d") {
      const gen =
        datasetId in DATASETS_1D_CLASSIFICATION
          ? DATASETS_1D_CLASSIFICATION[datasetId as keyof typeof DATASETS_1D_CLASSIFICATION]
          : DATASETS_1D_REGRESSION[datasetId as keyof typeof DATASETS_1D_REGRESSION];
      if (!gen) return;
      const points = gen(80, 0);
      // Fit Y to the actual series — a fixed [-1.5, 1.5] clips linear/cubic hard.
      let yMin = Infinity;
      let yMax = -Infinity;
      for (const p of points) {
        if (p.y < yMin) yMin = p.y;
        if (p.y > yMax) yMax = p.y;
      }
      if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMin === yMax) {
        yMin = -1.2;
        yMax = 1.2;
      } else {
        const pad = (yMax - yMin) * 0.12;
        yMin -= pad;
        yMax += pad;
      }
      // Keep zero in view when the series crosses / sits near it.
      yMin = Math.min(yMin, -0.15);
      yMax = Math.max(yMax, 0.15);
      const mapY = (y: number) => ((yMax - y) / (yMax - yMin)) * size;
      const isRegression = problemType === "regression";

      ctx.strokeStyle = "rgba(0,0,0,0.12)";
      ctx.beginPath();
      ctx.moveTo(0, mapY(0));
      ctx.lineTo(size, mapY(0));
      ctx.stroke();
      for (const p of points) {
        const px = ((p.x + 6) / 12) * size;
        const py = mapY(p.y);
        ctx.beginPath();
        ctx.arc(px, py, 2.4, 0, Math.PI * 2);
        // Regression has no classes — one neutral color. Classification keeps ±1 hues.
        ctx.fillStyle = isRegression
          ? "rgb(100, 116, 139)"
          : p.label > 0 ? CLASS_1_HEX : CLASS_0_HEX;
        ctx.fill();
      }
      return;
    }

    const gen = DATASETS[datasetId as DatasetId];
    if (!gen) return;
    const isRegression = problemType === "regression";
    if (isRegression) {
      // Continuous labels as sparse dots are unreadable in a tiny thumb —
      // paint a dense target-field heatmap instead (same palette as the canvas).
      const side = 48;
      const cell = size / side;
      const field = gen(side * side, 0);
      for (const p of field) {
        const { r, g, b } = valueToRgb(p.label);
        const px = ((p.x + 6) / 12) * size;
        const py = ((6 - p.y) / 12) * size;
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(px - cell / 2, py - cell / 2, cell + 0.6, cell + 0.6);
      }
      return;
    }
    const points = gen(120, 0);
    for (const p of points) {
      const px = ((p.x + 6) / 12) * size;
      const py = ((6 - p.y) / 12) * size;
      ctx.beginPath();
      ctx.arc(px, py, 2.8, 0, Math.PI * 2);
      ctx.fillStyle = p.label > 0 ? CLASS_1_HEX : CLASS_0_HEX;
      ctx.fill();
    }
  }, [datasetId, mode, problemType]);

  return (
    <div className={`dataset ${selected ? "selected" : ""}`}>
      <button
        type="button"
        className={`data-thumbnail-btn ${selected ? "selected" : ""}`}
        onClick={onSelect}
        aria-label={label}
        aria-pressed={selected}
      >
        <canvas ref={canvasRef} className="data-thumbnail" />
        {selected && <span className="data-thumbnail-check" aria-hidden="true">✓</span>}
      </button>
      <span className="dataset-label">{label}</span>
    </div>
  );
}
