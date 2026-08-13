import type { ReactNode } from "react";
import { LayerSpec } from "@ml-vis/core/cnn";
import { useCnnMessages, type CnnMessages } from "./messages";

export interface CnnArchitecturePanelProps {
  layers: LayerSpec[];
  selectedIndex: number | null;
  onSelectLayer: (index: number) => void;
  onSetFilters: (index: number, filters: number) => void;
  onSetKernelSize: (index: number, kernelSize: number) => void;
  onSetPoolKind: (index: number, poolKind: "max" | "avg") => void;
}

/** Conv filter count: step 2, capped at 16. */
const FILTER_OPTIONS = [2, 4, 6, 8, 10, 12, 14, 16];
const KERNEL_OPTIONS_2D = [3, 5, 7, 9];
const KERNEL_OPTIONS_1D = [3, 5, 7, 9];

function layerTitle(spec: LayerSpec, t: CnnMessages): string {
  switch (spec.kind) {
    case "conv2d":
    case "conv1d":
      return t.paletteConv;
    case "pool2d":
    case "pool1d":
      return t.palettePool;
    case "gap2d":
    case "gap1d":
      return t.paletteGap;
    case "flatten":
      return t.flatten;
    case "dense":
      return t.paletteDense;
  }
}

function nearestOption(options: number[], current: number): number {
  let best = options[0]!;
  let bestDist = Math.abs(best - current);
  for (const o of options) {
    const d = Math.abs(o - current);
    if (d < bestDist) {
      best = o;
      bestDist = d;
    }
  }
  return best;
}

function stepOption(options: number[], current: number, delta: -1 | 1): number {
  const snapped = nearestOption(options, current);
  const i = options.indexOf(snapped);
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
    <div className="nn-arch-stepper">
      <button
        type="button"
        className="nn-icon-btn nn-icon-btn--sm"
        disabled={atMin}
        onClick={() => onChange(stepOption(options, value, -1))}
        aria-label={ariaDec}
      >
        −
      </button>
      <span className="nn-arch-value">{value}</span>
      <button
        type="button"
        className="nn-icon-btn nn-icon-btn--sm"
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
  onSetFilters,
  onSetKernelSize,
  onSetPoolKind,
}: CnnArchitecturePanelProps) {
  const t = useCnnMessages();

  return (
    <div className="nn-arch-panel cnn-arch-panel">
      <h4 className="nn-flow-dock-title">{t.network}</h4>

      <div className="nn-arch-stack">
        {layers.map((spec, idx) => {
          const isConv = spec.kind === "conv2d" || spec.kind === "conv1d";
          const isPool = spec.kind === "pool2d" || spec.kind === "pool1d";
          const kernelOptions = spec.kind === "conv2d" ? KERNEL_OPTIONS_2D : KERNEL_OPTIONS_1D;
          const selected = selectedIndex === idx;
          const filterValue = nearestOption(FILTER_OPTIONS, spec.filters ?? 4);

          return (
            <div
              key={idx}
              className={`nn-arch-row cnn-arch-row${selected ? " selected" : ""}`}
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
              <span className="nn-arch-row-label">{layerTitle(spec, t)}</span>

              <div className="cnn-arch-row__controls">
                {isConv && (
                  <>
                    <ParamGroup label={t.filters}>
                      <MiniStepper
                        value={filterValue}
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
                    <div className="nn-flat-switch" role="group" aria-label={t.poolKind}>
                      <button
                        type="button"
                        className={`nn-flat-switch__btn${(spec.poolKind ?? "max") === "max" ? " selected" : ""}`}
                        onClick={() => onSetPoolKind(idx, "max")}
                      >
                        {t.poolMax}
                      </button>
                      <button
                        type="button"
                        className={`nn-flat-switch__btn${spec.poolKind === "avg" ? " selected" : ""}`}
                        onClick={() => onSetPoolKind(idx, "avg")}
                      >
                        {t.poolAvg}
                      </button>
                    </div>
                  </ParamGroup>
                )}

              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
