import type { LayerSpec } from "@ml-vis/core";
import { useCnnMessages } from "./messages";

export interface CnnArchitecturePanelProps {
  layers: LayerSpec[];
  selectedIndex: number | null;
  onSelectLayer: (index: number) => void;
  onAddLayer: (kind: "conv" | "pool" | "dense") => void;
  onRemoveLayer: (index: number) => void;
  onSetFilters: (index: number, filters: number) => void;
  onSetKernelSize: (index: number, kernelSize: number) => void;
  onSetUnits: (index: number, units: number) => void;
  onSetPoolKind: (index: number, poolKind: "max" | "avg") => void;
}

const FILTER_OPTIONS = [1, 2, 4, 8];
const KERNEL_OPTIONS_2D = [3, 5];
const KERNEL_OPTIONS_1D = [3, 5, 7];
const UNIT_OPTIONS = [1, 2, 4, 8];

function layerTitle(spec: LayerSpec, t: ReturnType<typeof useCnnMessages>): string {
  switch (spec.kind) {
    case "conv2d":
      return `${t.paletteConv}2D`;
    case "conv1d":
      return `${t.paletteConv}1D`;
    case "pool2d":
      return `${t.palettePool}2D`;
    case "pool1d":
      return `${t.palettePool}1D`;
    case "flatten":
      return t.network;
    case "dense":
      return t.paletteDense;
  }
}

/** Per-layer editor: shows controls for filters / kernel / units / pooling. */
export function CnnArchitecturePanel({
  layers,
  selectedIndex,
  onSelectLayer,
  onAddLayer,
  onRemoveLayer,
  onSetFilters,
  onSetKernelSize,
  onSetUnits,
  onSetPoolKind,
}: CnnArchitecturePanelProps) {
  const t = useCnnMessages();

  return (
    <div className="cnn-arch-panel">
      <h4 className="tf-flow-dock-title">{t.network}</h4>
      <div className="tf-arch-stack">
        {layers.map((spec, idx) => {
          const isConv = spec.kind === "conv2d" || spec.kind === "conv1d";
          const isPool = spec.kind === "pool2d" || spec.kind === "pool1d";
          const isDense = spec.kind === "dense";
          const kernelOptions = spec.kind === "conv2d" ? KERNEL_OPTIONS_2D : KERNEL_OPTIONS_1D;
          const selected = selectedIndex === idx;
          return (
            <div
              key={idx}
              className={`tf-arch-row${selected ? " selected" : ""}`}
              onClick={() => onSelectLayer(idx)}
              role="button"
              tabIndex={0}
            >
              <span className="tf-arch-row-label">{layerTitle(spec, t)}</span>
              <div className="cnn-arch-controls">
                {isConv && (
                  <>
                    <label className="cnn-arch-field">
                      <span>{t.filters}</span>
                      <select value={spec.filters ?? 4} onChange={(e) => onSetFilters(idx, Number(e.target.value))}>
                        {FILTER_OPTIONS.map((n) => (
                          <option key={n} value={n}>{n}</option>
                        ))}
                      </select>
                    </label>
                    <label className="cnn-arch-field">
                      <span>{t.kernelSize}</span>
                      <select value={spec.kernelSize ?? 3} onChange={(e) => onSetKernelSize(idx, Number(e.target.value))}>
                        {kernelOptions.map((n) => (
                          <option key={n} value={n}>{n}</option>
                        ))}
                      </select>
                    </label>
                  </>
                )}
                {isPool && (
                  <label className="cnn-arch-field">
                    <span>{t.poolKind}</span>
                    <select
                      value={spec.poolKind ?? "max"}
                      onChange={(e) => onSetPoolKind(idx, e.target.value as "max" | "avg")}
                    >
                      <option value="max">{t.poolMax}</option>
                      <option value="avg">{t.poolAvg}</option>
                    </select>
                  </label>
                )}
                {isDense && (
                  <label className="cnn-arch-field">
                    <span>{t.units}</span>
                    <select value={spec.units ?? 1} onChange={(e) => onSetUnits(idx, Number(e.target.value))}>
                      {UNIT_OPTIONS.map((n) => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </label>
                )}
                <button
                  type="button"
                  className="tf-icon-btn tf-icon-btn--sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveLayer(idx);
                  }}
                  aria-label={t.removeLayer}
                >
                  −
                </button>
              </div>
            </div>
          );
        })}
        <div className="cnn-arch-add">
          <button type="button" className="tf-btn tf-btn--secondary tf-btn--sm" onClick={() => onAddLayer("conv")}>
            + {t.paletteConv}
          </button>
          <button type="button" className="tf-btn tf-btn--secondary tf-btn--sm" onClick={() => onAddLayer("pool")}>
            + {t.palettePool}
          </button>
          <button type="button" className="tf-btn tf-btn--secondary tf-btn--sm" onClick={() => onAddLayer("dense")}>
            + {t.paletteDense}
          </button>
        </div>
      </div>
    </div>
  );
}
