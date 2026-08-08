import { useCallback } from "react";
import { getSignalPresetSpec, type SignalParamId } from "@ml-vis/core";
import type { SignalInput, SignalPresetId } from "./config";
import { useSignalMessages } from "./messages";

interface SignalControlsProps {
  /** Which input ("f" signal or "g" kernel) these controls edit. */
  target: "f" | "g";
  value: SignalInput;
  onChange: (next: SignalInput) => void;
}

/**
 * Preset picker + parameter sliders for one signal input. Reused for f and g on
 * every tab so the editing surface stays consistent.
 */
export function SignalControls({ target, value, onChange }: SignalControlsProps) {
  const t = useSignalMessages();
  const label = target === "f" ? t.signal : t.kernel;
  const spec = getSignalPresetSpec(value.preset);

  const onPresetChange = useCallback(
    (preset: SignalPresetId) => {
      const params: Partial<Record<SignalParamId, number>> = {};
      for (const p of getSignalPresetSpec(preset).params) params[p.id] = p.default;
      onChange({ preset, params });
    },
    [onChange],
  );

  const onParamChange = useCallback(
    (id: SignalParamId, v: number) => {
      onChange({ ...value, params: { ...value.params, [id]: v } });
    },
    [onChange, value],
  );

  return (
    <div className="signal-input">
      <div className="signal-input-head">
        <span className={`signal-input-tag signal-input-tag--${target}`}>{label}</span>
        <select
          className="nn-select nn-select--dock"
          value={value.preset}
          onChange={(e) => onPresetChange(e.target.value as SignalPresetId)}
        >
          {(["delta", "box", "gaussian", "triangle", "expDecay", "sinc", "cosine", "sine"] as SignalPresetId[]).map(
            (id) => (
              <option key={id} value={id}>
                {t.presetLabels[id]}
              </option>
            ),
          )}
        </select>
      </div>
      {spec.params.length > 0 && (
        <div className="signal-input-params">
          {spec.params.map((p) => (
            <label key={p.id} className="signal-slider">
              <span className="signal-slider-label">{t.paramLabels[p.id]}</span>
              <input
                type="range"
                min={p.min}
                max={p.max}
                step={p.step}
                value={value.params[p.id] ?? p.default}
                onChange={(e) => onParamChange(p.id, Number(e.target.value))}
              />
              <span className="signal-slider-value">
                {(value.params[p.id] ?? p.default).toFixed(p.step < 1 ? 1 : 0)}
              </span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
