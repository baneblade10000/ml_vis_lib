import type { CSSProperties } from "react";
import { useNetworkMessages } from "./messages";

export interface NetworkDataPanelProps {
  batchSize: number;
  noise: number;
  percTrainData: number;
  onBatchSizeChange: (size: number) => void;
  onNoiseChange: (noise: number) => void;
  onTrainRatioChange: (ratio: number) => void;
  onRegenerateData: () => void;
}

/** Data-generation controls — lives next to the dataset picker. */
export function NetworkDataPanel({
  batchSize,
  noise,
  percTrainData,
  onBatchSizeChange,
  onNoiseChange,
  onTrainRatioChange,
  onRegenerateData,
}: NetworkDataPanelProps) {
  const t = useNetworkMessages();

  return (
    <div className="tf-data-panel">
      <div className="tf-slider-group tf-slider-group--dock">
        <div className="tf-slider">
          <label>
            <span className="tf-slider-header">
              <span className="tf-slider-name">{t.noise}</span>
              <span className="value">{noise}</span>
            </span>
            <input
              type="range"
              min={0}
              max={50}
              step={1}
              value={noise}
              style={{ "--range-progress": `${(noise / 50) * 100}%` } as CSSProperties}
              onChange={(e) => onNoiseChange(Number(e.target.value))}
            />
          </label>
        </div>
        <div className="tf-slider">
          <label>
            <span className="tf-slider-header">
              <span className="tf-slider-name">{t.trainRatio}</span>
              <span className="value">{percTrainData}%</span>
            </span>
            <input
              type="range"
              min={10}
              max={90}
              step={5}
              value={percTrainData}
              style={{ "--range-progress": `${((percTrainData - 10) / 80) * 100}%` } as CSSProperties}
              onChange={(e) => onTrainRatioChange(Number(e.target.value))}
            />
          </label>
        </div>
        <div className="tf-slider">
          <label>
            <span className="tf-slider-header">
              <span className="tf-slider-name">{t.batchSize}</span>
              <span className="value">{batchSize}</span>
            </span>
            <input
              type="range"
              min={1}
              max={30}
              step={1}
              value={batchSize}
              style={{ "--range-progress": `${((batchSize - 1) / 29) * 100}%` } as CSSProperties}
              onChange={(e) => onBatchSizeChange(Number(e.target.value))}
            />
          </label>
        </div>
      </div>

      <button type="button" className="tf-btn tf-btn--secondary tf-data-regen" onClick={onRegenerateData}>
        <span className="tf-data-regen-icon" aria-hidden="true">
          ↻
        </span>
        {t.regenerateData}
      </button>
    </div>
  );
}
