import { PLAYGROUND_OPTIMIZERS, NETWORK_ACTIVATIONS, NETWORK_REGULARIZATION_RATES, NETWORK_REGULARIZATIONS, WEIGHT_INITS, type PlaygroundOptimizerId, type NetworkActivationId, type NetworkRegularizationId, type WeightInitId } from "@ml-vis/core/network";
import { useNetworkMessages } from "./messages";

const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];

export interface NetworkTrainingPanelProps {
  learningRate: number;
  optimizer: PlaygroundOptimizerId;
  activation: NetworkActivationId;
  weightInit: WeightInitId;
  regularization: NetworkRegularizationId;
  regularizationRate: number;
  discretize: boolean;
  onLearningRateChange: (rate: number) => void;
  onOptimizerChange: (optimizer: PlaygroundOptimizerId) => void;
  onActivationChange: (activation: NetworkActivationId) => void;
  onWeightInitChange: (init: WeightInitId) => void;
  onRegularizationChange: (regularization: NetworkRegularizationId) => void;
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
            {LEARNING_RATES.map((v) => (
              <option key={v} value={v}>
                {v}
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
            onChange={(e) => onActivationChange(e.target.value as NetworkActivationId)}
          >
            {Object.keys(NETWORK_ACTIVATIONS).map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>

        <label className="network-training-field">
          <span className="network-training-label">{t.weightInit}</span>
          <select
            className="nn-select nn-select--dock"
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

        <label className="network-training-field">
          <span className="network-training-label">{t.regularization}</span>
          <select
            className="nn-select nn-select--dock"
            value={regularization}
            onChange={(e) => onRegularizationChange(e.target.value as NetworkRegularizationId)}
          >
            {NETWORK_REGULARIZATIONS.map((id) => (
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
            {NETWORK_REGULARIZATION_RATES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>

        <label className="network-training-toggle">
          <span className="nn-toggle">
            <input
              type="checkbox"
              checked={discretize}
              onChange={(e) => onDiscretizeChange(e.target.checked)}
            />
            <span className="nn-toggle-track" />
          </span>
          {t.discretize}
        </label>
      </div>
    </div>
  );
}
