import type { CSSProperties } from "react";
import { useCnnMessages } from "./messages";

export interface CnnDataPanelProps {
  batchSize: number;
  noise: number;
  percTrainData: number;
  onBatchSizeChange: (size: number) => void;
  onNoiseChange: (noise: number) => void;
  onTrainRatioChange: (ratio: number) => void;
  onRegenerateData: () => void;
}

/** Data-generation controls — lives next to the dataset picker (MLP parity). */
export function CnnDataPanel({
  batchSize,
  noise,
  percTrainData,
  onBatchSizeChange,
  onNoiseChange,
  onTrainRatioChange,
  onRegenerateData,
}: CnnDataPanelProps) {
  const t = useCnnMessages();

  return (
    <div className="nn-data-panel">
      <div className="nn-slider-group nn-slider-group--dock">
        <div className="nn-slider">
          <div className="nn-slider-header">
            <span className="nn-slider-name">{t.noise}</span>
            <span className="value">{noise.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.02}
            value={noise}
            onChange={(e) => onNoiseChange(Number(e.target.value))}
            style={{ "--range-progress": `${(noise / 0.5) * 100}%` } as CSSProperties}
          />
        </div>
        <div className="nn-slider">
          <div className="nn-slider-header">
            <span className="nn-slider-name">{t.trainRatio}</span>
            <span className="value">{percTrainData}%</span>
          </div>
          <input
            type="range"
            min={10}
            max={90}
            step={5}
            value={percTrainData}
            onChange={(e) => onTrainRatioChange(Number(e.target.value))}
            style={{ "--range-progress": `${((percTrainData - 10) / 80) * 100}%` } as CSSProperties}
          />
        </div>
        <div className="nn-slider">
          <div className="nn-slider-header">
            <span className="nn-slider-name">{t.batchSize}</span>
            <span className="value">{batchSize}</span>
          </div>
          <input
            type="range"
            min={1}
            max={32}
            step={1}
            value={batchSize}
            onChange={(e) => onBatchSizeChange(Number(e.target.value))}
            style={{ "--range-progress": `${((batchSize - 1) / 31) * 100}%` } as CSSProperties}
          />
        </div>
      </div>
      <button type="button" className="nn-btn nn-btn--secondary nn-data-regen" onClick={onRegenerateData}>
        <span className="nn-data-regen-icon" aria-hidden>
          ↻
        </span>
        {t.regenerateData}
      </button>
    </div>
  );
}
