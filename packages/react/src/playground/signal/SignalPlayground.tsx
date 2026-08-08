import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSignalMessages, type SignalTabId } from "./messages";
import {
  buildConvolutionPayload,
  buildCorrelationPayload,
  buildFourierPayload,
  buildInputs,
  buildTheoremPayload,
  COLOR,
  DEFAULT_STATE,
  type SignalLabState,
} from "./config";
import { SignalControls } from "./SignalControls";
import { SignalCanvas } from "./SignalCanvas";

export interface SignalPlaygroundProps {
  initialConfig?: Partial<SignalLabState>;
  toolbarStart?: React.ReactNode;
  toolbarEnd?: React.ReactNode;
}

const TABS: SignalTabId[] = ["convolution", "correlation", "fourier", "theorem"];

/**
 * Signal lab: four tabs — convolution, cross-correlation, Fourier transform, and
 * the convolution theorem — sharing one editing surface (presets + parameter
 * sliders for f and g). The math is computed in `@ml-vis/core`; this component
 * owns UI state (active tab, presets, sliding position, play/pause) and renders
 * the active tab's payload via {@link SignalCanvas}.
 */
export function SignalPlayground({ initialConfig, toolbarStart, toolbarEnd }: SignalPlaygroundProps) {
  const t = useSignalMessages();
  const [state, setState] = useState<SignalLabState>({ ...DEFAULT_STATE, ...initialConfig });
  const [tab, setTab] = useState<SignalTabId>("convolution");
  const [playing, setPlaying] = useState(false);
  const rafRef = useRef<number | null>(null);

  const { f, g } = useMemo(() => buildInputs(state), [state]);

  // Reset the sliding position when the signal length changes so it stays valid.
  useEffect(() => {
    setState((prev) => (prev.position === 0 ? prev : { ...prev, position: 0 }));
  }, [state.length]);

  // Play/pause the sliding-kernel animation (convolution / correlation tabs).
  useEffect(() => {
    if (!playing || (tab !== "convolution" && tab !== "correlation")) return;
    const outLen = f.length + g.length - 1;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setState((prev) => {
        const next = prev.position + dt * Math.max(outLen / 4, 6);
        return { ...prev, position: next >= outLen - 1 ? 0 : next };
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [playing, tab, f.length, g.length]);

  const onTogglePlay = useCallback(() => setPlaying((p) => !p), []);
  const onStep = useCallback(
    () =>
      setState((prev) => {
        const outLen = f.length + g.length - 1;
        const next = Math.min(outLen - 1, Math.floor(prev.position) + 1);
        return { ...prev, position: next >= outLen - 1 ? 0 : next };
      }),
    [f.length, g.length],
  );
  const onReset = useCallback(() => {
    setState({ ...DEFAULT_STATE, ...initialConfig });
    setPlaying(false);
  }, [initialConfig]);

  const outLen = f.length + g.length - 1;
  const tabLabels: Record<SignalTabId, string> = {
    convolution: t.tabConvolution,
    correlation: t.tabCorrelation,
    fourier: t.tabFourier,
    theorem: t.tabTheorem,
  };
  const hint: Record<SignalTabId, string> = {
    convolution: t.convHint,
    correlation: t.corrHint,
    fourier: t.fourierHint,
    theorem: t.theoremHint,
  };

  const payload = useMemo(() => {
    switch (tab) {
      case "convolution":
        return buildConvolutionPayload(state, f, g, {
          f: t.legendF,
          g: t.legendKernel,
          result: t.legendConvolution,
          position: t.position,
        });
      case "correlation":
        return buildCorrelationPayload(state, f, g, {
          f: t.legendF,
          g: t.legendKernel,
          correlation: t.legendCorrelation,
          convolution: t.legendConvolution,
          position: t.position,
        });
      case "fourier":
        return buildFourierPayload(state, f, {
          f: t.legendF,
          spectrum: t.legendSpectrum,
          harmonic: t.legendHarmonic,
        });
      case "theorem":
        return buildTheoremPayload(f, g, {
          f: t.legendF,
          g: t.legendKernel,
          direct: t.legendDirect,
          fft: t.legendFft,
          recipe: t.recipe,
          deviation: t.maxDeviation,
        });
    }
  }, [tab, state, f, g, t]);

  return (
    <div className="signal-playground">
      <div className="signal-toolbar">
        <div className="signal-toolbar-left">
          {toolbarStart}
          <button type="button" className="nn-btn nn-btn--ghost" onClick={onReset}>
            {t.reset}
          </button>
        </div>
        <div className="signal-toolbar-right">{toolbarEnd}</div>
      </div>

      <div className="signal-tabs" role="tablist">
        {TABS.map((id) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            className={`signal-tab${tab === id ? " signal-tab--active" : ""}`}
            style={tab === id ? { borderColor: COLOR.f } : undefined}
            onClick={() => setTab(id)}
          >
            {tabLabels[id]}
          </button>
        ))}
      </div>

      <div className="signal-body">
        <div className="signal-canvas-wrap">
          <SignalCanvas payload={payload} />
          <p className="signal-hint">{hint[tab]}</p>
        </div>

        <aside className="signal-side">
          <div className="signal-inputs">
            <SignalControls
              target="f"
              value={state.f}
              onChange={(f) => setState((prev) => ({ ...prev, f }))}
            />
            <SignalControls
              target="g"
              value={state.g}
              onChange={(g) => setState((prev) => ({ ...prev, g }))}
            />
          </div>

          {tab === "convolution" || tab === "correlation" ? (
            <div className="signal-playback">
              <button
                type="button"
                className="nn-btn nn-btn--primary"
                onClick={onTogglePlay}
              >
                {playing ? t.pause : t.play}
              </button>
              <button type="button" className="nn-btn nn-btn--ghost" onClick={onStep}>
                {t.step}
              </button>
              <label className="signal-slider signal-slider--wide">
                <span className="signal-slider-label">{t.position}</span>
                <input
                  type="range"
                  min={0}
                  max={outLen - 1}
                  step={1}
                  value={Math.round(state.position)}
                  onChange={(e) => {
                    setPlaying(false);
                    setState((prev) => ({ ...prev, position: Number(e.target.value) }));
                  }}
                />
                <span className="signal-slider-value">{Math.round(state.position)}</span>
              </label>
            </div>
          ) : null}

          {tab === "fourier" ? (
            <label className="signal-slider signal-slider--wide">
              <span className="signal-slider-label">{t.harmonic}</span>
              <input
                type="range"
                min={0}
                max={f.length - 1}
                step={1}
                value={state.harmonic}
                onChange={(e) =>
                  setState((prev) => ({ ...prev, harmonic: Number(e.target.value) }))
                }
              />
              <span className="signal-slider-value">{state.harmonic}</span>
            </label>
          ) : null}

          {tab === "correlation" ? (
            <label className="signal-toggle">
              <input
                type="checkbox"
                checked={state.correlationShowConvolution}
                onChange={(e) =>
                  setState((prev) => ({
                    ...prev,
                    correlationShowConvolution: e.target.checked,
                  }))
                }
              />
              <span>{t.showAsConvolution}</span>
            </label>
          ) : null}
        </aside>
      </div>
    </div>
  );
}
