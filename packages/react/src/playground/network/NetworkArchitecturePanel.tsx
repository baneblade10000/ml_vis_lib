import { useNetworkMessages } from "./messages";

export interface NetworkArchitecturePanelProps {
  numHiddenLayers: number;
  networkShape: number[];
  maxHiddenLayers?: number;
  maxNeuronsPerLayer?: number;
  onAddLayer: () => void;
  onRemoveLayer: () => void;
  onAddNeuron: (layerIdx: number) => void;
  onRemoveNeuron: (layerIdx: number) => void;
}

export function NetworkArchitecturePanel({
  numHiddenLayers,
  networkShape,
  maxHiddenLayers = 6,
  maxNeuronsPerLayer = 8,
  onAddLayer,
  onRemoveLayer,
  onAddNeuron,
  onRemoveNeuron,
}: NetworkArchitecturePanelProps) {
  const t = useNetworkMessages();

  return (
    <div className="tf-arch-panel">
      <h4 className="tf-flow-dock-title">{t.network}</h4>

      <div className="tf-arch-stack">
        <div className="tf-arch-row tf-arch-row--header">
          <span className="tf-arch-row-label">{t.hiddenLayers}</span>
          <div className="tf-arch-stepper">
            <button
              type="button"
              className="tf-icon-btn tf-icon-btn--sm"
              onClick={onRemoveLayer}
              disabled={numHiddenLayers <= 0}
              aria-label={t.removeHiddenLayer}
            >
              −
            </button>
            <span className="tf-arch-value">{numHiddenLayers}</span>
            <button
              type="button"
              className="tf-icon-btn tf-icon-btn--sm"
              onClick={onAddLayer}
              disabled={numHiddenLayers >= maxHiddenLayers}
              aria-label={t.addHiddenLayer}
            >
              +
            </button>
          </div>
        </div>

        {Array.from({ length: numHiddenLayers }).map((_, layerIdx) => {
          const count = networkShape[layerIdx] ?? 0;
          return (
            <div key={layerIdx} className="tf-arch-row">
              <span className="tf-arch-row-label">{`${t.layer} ${layerIdx + 1}`}</span>
              <div className="tf-arch-stepper">
                <button
                  type="button"
                  className="tf-icon-btn tf-icon-btn--sm"
                  onClick={() => onRemoveNeuron(layerIdx)}
                  disabled={count <= 1}
                  aria-label={`${t.removeNeuron} ${layerIdx + 1}`}
                >
                  −
                </button>
                <span className="tf-arch-value">{count}</span>
                <button
                  type="button"
                  className="tf-icon-btn tf-icon-btn--sm"
                  onClick={() => onAddNeuron(layerIdx)}
                  disabled={count >= maxNeuronsPerLayer}
                  aria-label={`${t.addNeuron} ${layerIdx + 1}`}
                >
                  +
                </button>
              </div>
            </div>
          );
        })}

        {numHiddenLayers === 0 && <p className="tf-arch-hint">{t.noHiddenLayers}</p>}
      </div>
    </div>
  );
}
