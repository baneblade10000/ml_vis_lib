import type { ReactNode } from "react";
import type { LayerSpec } from "@ml-vis/core";
import { useCnnMessages, type CnnMessages } from "./messages";

export interface CnnArchitecturePanelProps {
  layers: LayerSpec[];
  selectedIndex: number | null;
  onSelectLayer: (index: number) => void;
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

function layerTitle(spec: LayerSpec, t: CnnMessages): string {
  switch (spec.kind) {
    case "conv2d":
    case "conv1d":
      return t.paletteConv;
    case "pool2d":
    case "pool1d":
      return t.palettePool;
    case "flatten":
      return t.flatten;
    case "dense":
      return t.paletteDense;
  }
}

function stepOption(options: number[], current: number, delta: -1 | 1): number {
  const i = Math.max(0, options.indexOf(current));
  return options[Math.max(0, Math.min(options.length - 1, i + delta))]!;
}

function ParamGroup({
  label,
  children,
}: {
  label?: string;
  children: ReactNode;
}) {
  return (
    <div className="cnn-arch-param" onClick={(e) => e.stopPropagation()}>
      <span className={`cnn-arch-param__label${label ? "" : " cnn-arch-param__label--empty"}`}>
        {label ?? "\u00a0"}
      </span>
      <div className="cnn-arch-param__control">{children}</div>
    </div>
  );
}

function MiniStepper({
  value,
  options,
  onChange,
  ariaDec,
  ariaInc,
}: {
  value: number;
  options: number[];
  onChange: (v: number) => void;
  ariaDec: string;
  ariaInc: string;
}) {
  const atMin = value === options[0];
  const atMax = value === options[options.length - 1];
  return (
    <div className="tf-arch-stepper">
      <button
        type="button"
        className="tf-icon-btn tf-icon-btn--sm"
        disabled={atMin}
        onClick={() => onChange(stepOption(options, value, -1))}
        aria-label={ariaDec}
      >
        −
      </button>
      <span className="tf-arch-value">{value}</span>
      <button
        type="button"
        className="tf-icon-btn tf-icon-btn--sm"
        disabled={atMax}
        onClick={() => onChange(stepOption(options, value, 1))}
        aria-label={ariaInc}
      >
        +
      </button>
    </div>
  );
}

/** Compact layer editor: label left, named params with steppers right. */
export function CnnArchitecturePanel({
  layers,
  selectedIndex,
  onSelectLayer,
  onRemoveLayer,
  onSetFilters,
  onSetKernelSize,
  onSetUnits,
  onSetPoolKind,
}: CnnArchitecturePanelProps) {
  const t = useCnnMessages();

  return (
    <div className="tf-arch-panel cnn-arch-panel">
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
              className={`tf-arch-row cnn-arch-row${selected ? " selected" : ""}`}
              onClick={() => onSelectLayer(idx)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelectLayer(idx);
                }
              }}
              role="button"
              tabIndex={0}
              aria-pressed={selected}
            >
              <span className="tf-arch-row-label">{layerTitle(spec, t)}</span>

              <div className="cnn-arch-row__controls">
                {isConv && (
                  <>
                    <ParamGroup label={t.filters}>
                      <MiniStepper
                        value={spec.filters ?? 4}
                        options={FILTER_OPTIONS}
                        onChange={(v) => onSetFilters(idx, v)}
                        ariaDec={`${t.filters} −`}
                        ariaInc={`${t.filters} +`}
                      />
                    </ParamGroup>
                    <ParamGroup label={t.kernelSize}>
                      <MiniStepper
                        value={spec.kernelSize ?? 3}
                        options={kernelOptions}
                        onChange={(v) => onSetKernelSize(idx, v)}
                        ariaDec={`${t.kernelSize} −`}
                        ariaInc={`${t.kernelSize} +`}
                      />
                    </ParamGroup>
                  </>
                )}

                {isPool && (
                  <ParamGroup>
                    <div className="tf-flat-switch" role="group" aria-label={t.poolKind}>
                      <button
                        type="button"
                        className={`tf-flat-switch__btn${(spec.poolKind ?? "max") === "max" ? " selected" : ""}`}
                        onClick={() => onSetPoolKind(idx, "max")}
                      >
                        {t.poolMax}
                      </button>
                      <button
                        type="button"
                        className={`tf-flat-switch__btn${spec.poolKind === "avg" ? " selected" : ""}`}
                        onClick={() => onSetPoolKind(idx, "avg")}
                      >
                        {t.poolAvg}
                      </button>
                    </div>
                  </ParamGroup>
                )}

                {isDense && (
                  <ParamGroup label={t.units}>
                    <MiniStepper
                      value={spec.units ?? 1}
                      options={UNIT_OPTIONS}
                      onChange={(v) => onSetUnits(idx, v)}
                      ariaDec={`${t.units} −`}
                      ariaInc={`${t.units} +`}
                    />
                  </ParamGroup>
                )}

                <ParamGroup>
                  <button
                    type="button"
                    className="tf-icon-btn tf-icon-btn--sm"
                    onClick={() => onRemoveLayer(idx)}
                    aria-label={t.removeLayer}
                  >
                    −
                  </button>
                </ParamGroup>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
