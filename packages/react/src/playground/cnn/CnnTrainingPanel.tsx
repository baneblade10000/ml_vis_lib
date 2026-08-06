import type { CnnActivationId } from "@ml-vis/core";
import { useCnnMessages } from "./messages";

const LEARNING_RATES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3];
const ACTIVATIONS: CnnActivationId[] = ["relu", "tanh", "sigmoid", "linear"];

export interface CnnTrainingPanelProps {
  learningRate: number;
  activation: CnnActivationId;
  batchSize: number;
  noise: number;
  percTrainData: number;
  onLearningRateChange: (lr: number) => void;
  onActivationChange: (a: CnnActivationId) => void;
  onBatchSizeChange: (bs: number) => void;
  onNoiseChange: (n: number) => void;
  onTrainRatioChange: (r: number) => void;
  onRegenerateData: () => void;
}

/** Training hyperparameter controls — mirrors the network training panel. */
export function CnnTrainingPanel({
  learningRate,
  activation,
  batchSize,
  noise,
  percTrainData,
  onLearningRateChange,
  onActivationChange,
  onBatchSizeChange,
  onNoiseChange,
  onTrainRatioChange,
  onRegenerateData,
}: CnnTrainingPanelProps) {
  const t = useCnnMessages();

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
            {LEARNING_RATES.map((lr) => (
              <option key={lr} value={lr}>{lr}</option>
            ))}
          </select>
        </label>
        <label className="tf-training-field">
          <span className="tf-training-label">{t.activation}</span>
          <select
            className="tf-select tf-select--dock"
            value={activation}
            onChange={(e) => onActivationChange(e.target.value as CnnActivationId)}
          >
            {ACTIVATIONS.map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="tf-slider-group tf-slider-group--dock">
        <div className="tf-slider">
          <div className="tf-slider-header">
            <span className="tf-slider-name">{t.noise}</span>
            <span className="value">{noise.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.02}
            value={noise}
            onChange={(e) => onNoiseChange(Number(e.target.value))}
            style={{ "--range-progress": `${(noise / 0.5) * 100}%` } as React.CSSProperties}
          />
        </div>
        <div className="tf-slider">
          <div className="tf-slider-header">
            <span className="tf-slider-name">{t.trainRatio}</span>
            <span className="value">{percTrainData}%</span>
          </div>
          <input
            type="range"
            min={10}
            max={90}
            step={5}
            value={percTrainData}
            onChange={(e) => onTrainRatioChange(Number(e.target.value))}
            style={{ "--range-progress": `${((percTrainData - 10) / 80) * 100}%` } as React.CSSProperties}
          />
        </div>
        <div className="tf-slider">
          <div className="tf-slider-header">
            <span className="tf-slider-name">{t.batchSize}</span>
            <span className="value">{batchSize}</span>
          </div>
          <input
            type="range"
            min={1}
            max={32}
            step={1}
            value={batchSize}
            onChange={(e) => onBatchSizeChange(Number(e.target.value))}
            style={{ "--range-progress": `${((batchSize - 1) / 31) * 100}%` } as React.CSSProperties}
          />
        </div>
      </div>
      <button type="button" className="tf-btn tf-btn--secondary tf-data-regen" onClick={onRegenerateData}>
        <span className="tf-data-regen-icon" aria-hidden>↻</span>
        {t.regenerateData}
      </button>
    </div>
  );
}
