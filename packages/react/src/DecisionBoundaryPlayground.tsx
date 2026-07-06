import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DecisionBoundaryPlot,
  DEFAULT_CONFIG,
  type DecisionBoundaryPayload,
  type LiveTrainingState,
  type PlaygroundConfig,
  advanceLiveTraining,
  buildPayload,
  configSignature,
  createLiveTrainingState,
  frameIndexForEpoch,
  resetLiveTraining,
} from "@ml-vis/core";
import { ChartBox, useCanvasChart } from "./useCanvasChart";

export interface DecisionBoundaryChartProps {
  state: LiveTrainingState;
  frameIndex?: number;
  height?: number;
  onHover?: (info: { x1: number; x2: number; probability: number } | null) => void;
}

function payloadForFrame(state: LiveTrainingState, frameIndex: number): DecisionBoundaryPayload {
  const payload = buildPayload(state);
  const snapshots = payload.snapshots;
  const index = Math.max(0, Math.min(frameIndex, Math.max(snapshots.length - 1, 0)));
  const frame = snapshots[index];
  const previous = snapshots[Math.max(0, index - 1)];
  return {
    samples: payload.samples,
    grid: frame?.grid ?? payload.grid,
    previousGrid: previous?.grid,
    history: payload.history,
    epoch: frame?.epoch ?? state.epoch,
    frameLabel: frame ? `epoch ${frame.epoch}` : `epoch ${state.epoch}`,
  };
}

export function DecisionBoundaryChart({
  state,
  frameIndex,
  height = 420,
  onHover,
}: DecisionBoundaryChartProps) {
  const plotPayload = useMemo(
    () => payloadForFrame(state, frameIndex ?? frameIndexForEpoch(buildPayload(state).snapshots, state.epoch)),
    [state, frameIndex],
  );

  const plotRef = useRef<DecisionBoundaryPlot | null>(null);
  const [width, setWidth] = useState(0);

  const { canvasRef } = useCanvasChart(
    (canvas) => {
      const plot = new DecisionBoundaryPlot(canvas);
      plotRef.current = plot;
      return plot;
    },
    (plot) => plot.destroy(),
    [height],
    (plot) => {
      if (width > 0) plot.setSize(width, height);
      plot.setData(plotPayload);
    },
    [plotPayload, width, height],
  );

  const handleMove = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      plotRef.current?.setPointer({ clientX: event.clientX, clientY: event.clientY });
      onHover?.(plotRef.current?.nearestGridPoint(event.clientX, event.clientY) ?? null);
    },
    [onHover],
  );

  const handleLeave = useCallback(() => {
    plotRef.current?.setPointer(null);
    onHover?.(null);
  }, [onHover]);

  return (
    <ChartBox height={height} onWidth={setWidth}>
      {(measuredWidth) => (
        <canvas
          ref={canvasRef}
          style={{ width: measuredWidth, height, display: "block", cursor: "crosshair" }}
          onMouseMove={handleMove}
          onMouseLeave={handleLeave}
        />
      )}
    </ChartBox>
  );
}

export interface PlaygroundControlsProps {
  config: PlaygroundConfig;
  state: LiveTrainingState;
  frameIndex: number;
  playing: boolean;
  hover: { x1: number; x2: number; probability: number } | null;
  onConfigChange: (config: PlaygroundConfig) => void;
  onStateChange: (state: LiveTrainingState) => void;
  onFrameIndexChange: (index: number) => void;
  onPlayingChange: (playing: boolean) => void;
}

function parseHiddenLayers(raw: string): number[] {
  const values = raw
    .split(/[,\s;]+/)
    .map((token) => Number.parseInt(token.trim(), 10))
    .filter((value) => Number.isFinite(value) && value > 0);
  return values.length ? values : [8, 8];
}

export function PlaygroundControls({
  config,
  state,
  frameIndex,
  playing,
  hover,
  onConfigChange,
  onStateChange,
  onFrameIndexChange,
  onPlayingChange,
}: PlaygroundControlsProps) {
  const payload = useMemo(() => buildPayload(state), [state]);
  const latest = payload.history[payload.history.length - 1];
  const pending = configSignature(config) !== configSignature(state.config);

  const reset = () => {
    const next = resetLiveTraining(config);
    onStateChange(next);
    onFrameIndexChange(0);
    onPlayingChange(false);
  };

  const step = () => {
    const base = pending ? resetLiveTraining(config) : state;
    if (pending) onStateChange(base);
    const next = advanceLiveTraining({ ...base, playing: false }, 1);
    onStateChange(next);
    onFrameIndexChange(frameIndexForEpoch(buildPayload(next).snapshots, next.epoch));
  };

  const togglePlay = () => {
    if (!playing && pending) reset();
    onPlayingChange(!playing);
  };

  return (
    <div className="playground-controls">
      <div className="control-row">
        <label>
          Dataset
          <select
            value={config.dataset}
            onChange={(e) => onConfigChange({ ...config, dataset: e.target.value as PlaygroundConfig["dataset"] })}
          >
            <option value="circles">Circles</option>
            <option value="xor">XOR</option>
            <option value="spiral">Spiral</option>
            <option value="gaussian">Gaussian</option>
          </select>
        </label>
        <label>
          Activation
          <select
            value={config.activation}
            onChange={(e) => onConfigChange({ ...config, activation: e.target.value as PlaygroundConfig["activation"] })}
          >
            <option value="tanh">tanh</option>
            <option value="relu">relu</option>
            <option value="sigmoid">sigmoid</option>
          </select>
        </label>
        <label>
          Hidden layers
          <input
            type="text"
            value={config.hiddenLayers.join(",")}
            onChange={(e) => onConfigChange({ ...config, hiddenLayers: parseHiddenLayers(e.target.value) })}
          />
        </label>
      </div>

      <div className="control-row">
        <label>
          Noise {config.noise.toFixed(2)}
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.01}
            value={config.noise}
            onChange={(e) => onConfigChange({ ...config, noise: Number(e.target.value) })}
          />
        </label>
        <label>
          Learning rate {config.learningRate.toFixed(3)}
          <input
            type="range"
            min={0.001}
            max={0.1}
            step={0.001}
            value={config.learningRate}
            onChange={(e) => onConfigChange({ ...config, learningRate: Number(e.target.value) })}
          />
        </label>
        <label>
          Epochs {config.epochs}
          <input
            type="range"
            min={20}
            max={200}
            step={5}
            value={config.epochs}
            onChange={(e) => onConfigChange({ ...config, epochs: Number(e.target.value) })}
          />
        </label>
      </div>

      <div className="control-row control-actions">
        <button type="button" onClick={togglePlay}>
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={step}>
          Step +1
        </button>
        <button type="button" onClick={reset}>
          Reset & train
        </button>
        <span className="metric-pill">
          epoch {state.epoch}/{config.epochs}
        </span>
        <span className="metric-pill">
          val acc {(100 * (latest?.validationAccuracy ?? 0)).toFixed(1)}%
        </span>
        {hover && (
          <span className="metric-pill">
            p(class 1) {(100 * hover.probability).toFixed(1)}%
          </span>
        )}
        {pending && <span className="metric-pill pending">config changed — reset or play</span>}
      </div>

      <label className="scrubber-row">
        Replay frame {frameIndex + 1}/{Math.max(payload.snapshots.length, 1)}
        <input
          type="range"
          min={0}
          max={Math.max(payload.snapshots.length - 1, 0)}
          step={1}
          value={frameIndex}
          onChange={(e) => onFrameIndexChange(Number(e.target.value))}
        />
      </label>
    </div>
  );
}

export function DecisionBoundaryPlayground({ initialConfig }: { initialConfig?: PlaygroundConfig }) {
  const [config, setConfig] = useState<PlaygroundConfig>(initialConfig ?? DEFAULT_CONFIG);
  const [state, setState] = useState<LiveTrainingState>(() => createLiveTrainingState(initialConfig ?? DEFAULT_CONFIG));
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState<{ x1: number; x2: number; probability: number } | null>(null);
  const configRef = useRef(config);
  configRef.current = config;
  const pendingRef = useRef(false);
  pendingRef.current = configSignature(config) !== configSignature(state.config);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setState((current) => {
        let base = current;
        if (pendingRef.current) {
          base = resetLiveTraining(configRef.current);
          pendingRef.current = false;
        }
        if (base.epoch >= configRef.current.epochs) {
          setPlaying(false);
          return base;
        }
        const next = advanceLiveTraining({ ...base, playing: true }, 1);
        setFrameIndex(frameIndexForEpoch(buildPayload(next).snapshots, next.epoch));
        return next;
      });
    }, 60);
    return () => window.clearInterval(timer);
  }, [playing]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        event.preventDefault();
        setPlaying((value) => !value);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <section className="playground-panel">
      <DecisionBoundaryChart state={state} frameIndex={frameIndex} onHover={setHover} />
      <PlaygroundControls
        config={config}
        state={state}
        frameIndex={frameIndex}
        playing={playing}
        hover={hover}
        onConfigChange={setConfig}
        onStateChange={setState}
        onFrameIndexChange={setFrameIndex}
        onPlayingChange={setPlaying}
      />
      <p className="playground-hint">Space — play/pause. Hover — inspect probability. Scrubber — replay boundary snapshots.</p>
    </section>
  );
}
