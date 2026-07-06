import type { CSSProperties } from "react";
import { TF_ACTIVATIONS, type ArchitecturePresetId, type TfActivationId } from "@ml-vis/core";
import { useNetworkMessages } from "./messages";

const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];

export interface NetworkTrainingPanelProps {
  architecturePreset: ArchitecturePresetId;
  learningRate: number;
  activation: TfActivationId;
  batchSize: number;
  noise: number;
  percTrainData: number;
  discretize: boolean;
  onArchitecturePresetChange: (preset: ArchitecturePresetId) => void;
  onLearningRateChange: (rate: number) => void;
  onActivationChange: (activation: TfActivationId) => void;
  onBatchSizeChange: (size: number) => void;
  onNoiseChange: (noise: number) => void;
  onTrainRatioChange: (ratio: number) => void;
  onDiscretizeChange: (value: boolean) => void;
  onRegenerateData: () => void;
}

export function NetworkTrainingPanel({
  learningRate,
  activation,
  batchSize,
  noise,
  percTrainData,
  discretize,
  onLearningRateChange,
  onActivationChange,
  onBatchSizeChange,
  onNoiseChange,
  onTrainRatioChange,
  onDiscretizeChange,
  onRegenerateData,
}: NetworkTrainingPanelProps) {
  const t = useNetworkMessages();

  return (
    <div className="tf-training-panel">
      <h4 className="tf-flow-dock-title">{t.training}</h4>

      <div className="tf-training-fields">
        <label className="tf-training-field">
          <span className="tf-training-label">{t.learningRate}</span>
          <select
            className="tf-select tf-select--dock"
            value={learningRate}
            onChange={(e) => onLearningRateChange(Number(e.target.value))}
          >
            {LEARNING_RATES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>

        <label className="tf-training-field">
          <span className="tf-training-label">{t.activation}</span>
          <select
            className="tf-select tf-select--dock"
            value={activation}
            onChange={(e) => onActivationChange(e.target.value as TfActivationId)}
          >
            {Object.keys(TF_ACTIVATIONS).map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>

        <label className="tf-training-toggle">
          <span className="tf-toggle">
            <input
              type="checkbox"
              checked={discretize}
              onChange={(e) => onDiscretizeChange(e.target.checked)}
            />
            <span className="tf-toggle-track" />
          </span>
          {t.discretize}
        </label>
      </div>

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
