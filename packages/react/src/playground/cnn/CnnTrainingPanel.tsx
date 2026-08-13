import { CNN_REGULARIZATION_RATES, CNN_REGULARIZATIONS, PLAYGROUND_OPTIMIZERS, type CnnActivationId, type CnnRegularizationId, type PlaygroundOptimizerId } from "@ml-vis/core/cnn";
import { useCnnMessages } from "./messages";

const LEARNING_RATES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3];
const ACTIVATIONS: CnnActivationId[] = ["relu", "tanh", "sigmoid", "linear"];

export interface CnnTrainingPanelProps {
  learningRate: number;
  optimizer: PlaygroundOptimizerId;
  activation: CnnActivationId;
  regularization: CnnRegularizationId;
  regularizationRate: number;
  onLearningRateChange: (lr: number) => void;
  onOptimizerChange: (optimizer: PlaygroundOptimizerId) => void;
  onActivationChange: (a: CnnActivationId) => void;
  onRegularizationChange: (regularization: CnnRegularizationId) => void;
  onRegularizationRateChange: (rate: number) => void;
}

/** Training hyperparams — data sliders live with the dataset (MLP parity). */
export function CnnTrainingPanel({
  learningRate,
  optimizer,
  activation,
  regularization,
  regularizationRate,
  onLearningRateChange,
  onOptimizerChange,
  onActivationChange,
  onRegularizationChange,
  onRegularizationRateChange,
}: CnnTrainingPanelProps) {
  const t = useCnnMessages();

  return (
    <div className="network-training-panel">
      <h4 className="nn-flow-dock-title">{t.training}</h4>
      <div className="network-training-fields">
        <label className="network-training-field">
          <span className="network-training-label">{t.learningRate}</span>
          <select
            className="nn-select nn-select--dock"
            value={learningRate}
            onChange={(e) => onLearningRateChange(Number(e.target.value))}
          >
            {LEARNING_RATES.map((lr) => (
              <option key={lr} value={lr}>
                {lr}
              </option>
            ))}
          </select>
        </label>
        <label className="network-training-field">
          <span className="network-training-label">{t.optimizer}</span>
          <select
            className="nn-select nn-select--dock"
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
        <label className="network-training-field">
          <span className="network-training-label">{t.activation}</span>
          <select
            className="nn-select nn-select--dock"
            value={activation}
            onChange={(e) => onActivationChange(e.target.value as CnnActivationId)}
          >
            {ACTIVATIONS.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </label>
        <label className="network-training-field">
          <span className="network-training-label">{t.regularization}</span>
          <select
            className="nn-select nn-select--dock"
            value={regularization}
            onChange={(e) => onRegularizationChange(e.target.value as CnnRegularizationId)}
          >
            {CNN_REGULARIZATIONS.map((id) => (
              <option key={id} value={id}>
                {t.regularizationLabels[id]}
              </option>
            ))}
          </select>
        </label>
        <label className="network-training-field">
          <span className="network-training-label">{t.regularizationRate}</span>
          <select
            className="nn-select nn-select--dock"
            value={regularizationRate}
            disabled={regularization === "none"}
            onChange={(e) => onRegularizationRateChange(Number(e.target.value))}
          >
            {CNN_REGULARIZATION_RATES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
