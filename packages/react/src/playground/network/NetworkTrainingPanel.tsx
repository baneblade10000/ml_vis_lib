import {
  PLAYGROUND_OPTIMIZERS,
  TF_ACTIVATIONS,
  TF_REGULARIZATION_RATES,
  TF_REGULARIZATIONS,
  WEIGHT_INITS,
  type PlaygroundOptimizerId,
  type TfActivationId,
  type TfRegularizationId,
  type WeightInitId,
} from "@ml-vis/core";
import { useNetworkMessages } from "./messages";

const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];

export interface NetworkTrainingPanelProps {
  learningRate: number;
  optimizer: PlaygroundOptimizerId;
  activation: TfActivationId;
  weightInit: WeightInitId;
  regularization: TfRegularizationId;
  regularizationRate: number;
  discretize: boolean;
  onLearningRateChange: (rate: number) => void;
  onOptimizerChange: (optimizer: PlaygroundOptimizerId) => void;
  onActivationChange: (activation: TfActivationId) => void;
  onWeightInitChange: (init: WeightInitId) => void;
  onRegularizationChange: (regularization: TfRegularizationId) => void;
  onRegularizationRateChange: (rate: number) => void;
  onDiscretizeChange: (value: boolean) => void;
}

export function NetworkTrainingPanel({
  learningRate,
  optimizer,
  activation,
  weightInit,
  regularization,
  regularizationRate,
  discretize,
  onLearningRateChange,
  onOptimizerChange,
  onActivationChange,
  onWeightInitChange,
  onRegularizationChange,
  onRegularizationRateChange,
  onDiscretizeChange,
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
          <span className="tf-training-label">{t.optimizer}</span>
          <select
            className="tf-select tf-select--dock"
            value={optimizer}
            onChange={(e) => onOptimizerChange(e.target.value as PlaygroundOptimizerId)}
          >
            {PLAYGROUND_OPTIMIZERS.map((id) => (
              <option key={id} value={id}>
                {id}
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

        <label className="tf-training-field">
          <span className="tf-training-label">{t.weightInit}</span>
          <select
            className="tf-select tf-select--dock"
            value={weightInit}
            onChange={(e) => onWeightInitChange(e.target.value as WeightInitId)}
          >
            {WEIGHT_INITS.map((id) => (
              <option key={id} value={id}>
                {t.weightInitLabels[id]}
              </option>
            ))}
          </select>
        </label>

        <label className="tf-training-field">
          <span className="tf-training-label">{t.regularization}</span>
          <select
            className="tf-select tf-select--dock"
            value={regularization}
            onChange={(e) => onRegularizationChange(e.target.value as TfRegularizationId)}
          >
            {TF_REGULARIZATIONS.map((id) => (
              <option key={id} value={id}>
                {t.regularizationLabels[id]}
              </option>
            ))}
          </select>
        </label>

        <label className="tf-training-field">
          <span className="tf-training-label">{t.regularizationRate}</span>
          <select
            className="tf-select tf-select--dock"
            value={regularizationRate}
            disabled={regularization === "none"}
            onChange={(e) => onRegularizationRateChange(Number(e.target.value))}
          >
            {TF_REGULARIZATION_RATES.map((v) => (
              <option key={v} value={v}>
                {v}
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
    </div>
  );
}
