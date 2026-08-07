import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DecisionBoundaryPlot,
  DEFAULT_CONFIG,
  TrainWorkerClient,
  canUseTrainWorkers,
  frameIndexForEpoch,
  type DecisionBoundaryPayload,
  type MlpTrainSnapshot,
  type PlaygroundConfig,
  type PlaygroundPayload,
  type TrainSnapshot,
  advanceLiveTraining,
  buildPayload,
  configSignature,
  createLiveTrainingState,
  resetLiveTraining,
  type LiveTrainingState,
} from "@ml-vis/core";
import { createMlpTrainWorker } from "@ml-vis/core/workers/createWorkers";
import { ChartBox, useCanvasChart } from "./useCanvasChart";

export interface DecisionBoundaryChartProps {
  payload: PlaygroundPayload;
  epoch: number;
  frameIndex?: number;
  height?: number;
  onHover?: (info: { x1: number; x2: number; probability: number } | null) => void;
}

function payloadForFrame(
  payload: PlaygroundPayload,
  epoch: number,
  frameIndex: number,
): DecisionBoundaryPayload {
  const snapshots = payload.snapshots;
  const index = Math.max(0, Math.min(frameIndex, Math.max(snapshots.length - 1, 0)));
  const frame = snapshots[index];
  const previous = snapshots[Math.max(0, index - 1)];
  return {
    samples: payload.samples,
    grid: frame?.grid ?? payload.grid,
    previousGrid: previous?.grid,
    history: payload.history,
    epoch: frame?.epoch ?? epoch,
    frameLabel: frame ? `epoch ${frame.epoch}` : `epoch ${epoch}`,
  };
}

export function DecisionBoundaryChart({
  payload,
  epoch,
  frameIndex,
  height = 420,
  onHover,
}: DecisionBoundaryChartProps) {
  const plotPayload = useMemo(
    () =>
      payloadForFrame(
        payload,
        epoch,
        frameIndex ?? frameIndexForEpoch(payload.snapshots, epoch),
      ),
    [payload, epoch, frameIndex],
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
  payload: PlaygroundPayload;
  epoch: number;
  frameIndex: number;
  playing: boolean;
  hover: { x1: number; x2: number; probability: number } | null;
  pending: boolean;
  onConfigChange: (config: PlaygroundConfig) => void;
  onFrameIndexChange: (index: number) => void;
  onPlayingChange: (playing: boolean) => void;
  onReset: () => void;
  onStep: () => void;
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
  payload,
  epoch,
  frameIndex,
  playing,
  hover,
  pending,
  onConfigChange,
  onFrameIndexChange,
  onPlayingChange,
  onReset,
  onStep,
}: PlaygroundControlsProps) {
  const latest = payload.history[payload.history.length - 1];

  const togglePlay = () => {
    if (!playing && pending) onReset();
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
        <button type="button" onClick={onStep}>
          Step +1
        </button>
        <button type="button" onClick={onReset}>
          Reset & train
        </button>
        <span className="metric-pill">
          epoch {epoch}/{config.epochs}
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

function isMlpSnapshot(s: TrainSnapshot | null): s is MlpTrainSnapshot {
  return s?.kind === "mlp";
}

export function DecisionBoundaryPlayground({ initialConfig }: { initialConfig?: PlaygroundConfig }) {
  const [config, setConfig] = useState<PlaygroundConfig>(initialConfig ?? DEFAULT_CONFIG);
  const [payload, setPayload] = useState<PlaygroundPayload | null>(null);
  const [epoch, setEpoch] = useState(0);
  const [trainedConfig, setTrainedConfig] = useState<PlaygroundConfig>(initialConfig ?? DEFAULT_CONFIG);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState<{ x1: number; x2: number; probability: number } | null>(null);
  const configRef = useRef(config);
  configRef.current = config;
  const clientRef = useRef<TrainWorkerClient | null>(null);
  const fallbackStateRef = useRef<LiveTrainingState | null>(null);
  const pending = configSignature(config) !== configSignature(trainedConfig);

  const applySnap = useCallback((snap: MlpTrainSnapshot) => {
    setPayload(snap.payload);
    setEpoch(snap.epoch);
    setTrainedConfig(snap.config);
    setFrameIndex(frameIndexForEpoch(snap.payload.snapshots, snap.epoch));
    if (snap.epoch >= snap.config.epochs) setPlaying(false);
  }, []);

  useEffect(() => {
    const cfg = initialConfig ?? DEFAULT_CONFIG;
    if (!canUseTrainWorkers()) {
      const state = createLiveTrainingState(cfg);
      fallbackStateRef.current = state;
      setPayload(buildPayload(state));
      setEpoch(state.epoch);
      setTrainedConfig(state.config);
      return;
    }
    const client = new TrainWorkerClient({
      createWorker: createMlpTrainWorker,
      onTick: (s) => {
        if (isMlpSnapshot(s)) applySnap(s);
      },
      onError: (message) => console.error("[mlp train worker]", message),
    });
    clientRef.current = client;
    void client.init(structuredClone(cfg)).then((s) => {
      if (isMlpSnapshot(s)) applySnap(s);
    });
    return () => {
      client.dispose();
      clientRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const client = clientRef.current;
    if (client) {
      if (playing) {
        if (pending) client.rebuild("reset", structuredClone(configRef.current));
        client.play(16);
      } else {
        client.pause();
      }
      return;
    }
    // Fallback main-thread loop
    if (!playing) return;
    const timer = window.setInterval(() => {
      let base = fallbackStateRef.current;
      if (!base) return;
      if (configSignature(configRef.current) !== configSignature(base.config)) {
        base = resetLiveTraining(configRef.current);
      }
      if (base.epoch >= configRef.current.epochs) {
        setPlaying(false);
        fallbackStateRef.current = base;
        setPayload(buildPayload(base));
        setEpoch(base.epoch);
        return;
      }
      const next = advanceLiveTraining({ ...base, playing: true }, 1);
      fallbackStateRef.current = next;
      setPayload(buildPayload(next));
      setEpoch(next.epoch);
      setTrainedConfig(next.config);
      setFrameIndex(frameIndexForEpoch(buildPayload(next).snapshots, next.epoch));
    }, 60);
    return () => window.clearInterval(timer);
  }, [playing, pending]);

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

  const onReset = useCallback(() => {
    setPlaying(false);
    const client = clientRef.current;
    if (client) {
      client.rebuild("reset", structuredClone(configRef.current));
      return;
    }
    const next = resetLiveTraining(configRef.current);
    fallbackStateRef.current = next;
    setPayload(buildPayload(next));
    setEpoch(0);
    setTrainedConfig(next.config);
    setFrameIndex(0);
  }, []);

  const onStep = useCallback(() => {
    setPlaying(false);
    const client = clientRef.current;
    if (client) {
      if (pending) client.rebuild("reset", structuredClone(configRef.current));
      client.step();
      return;
    }
    let base = fallbackStateRef.current ?? createLiveTrainingState(configRef.current);
    if (pending) base = resetLiveTraining(configRef.current);
    const next = advanceLiveTraining({ ...base, playing: false }, 1);
    fallbackStateRef.current = next;
    setPayload(buildPayload(next));
    setEpoch(next.epoch);
    setTrainedConfig(next.config);
    setFrameIndex(frameIndexForEpoch(buildPayload(next).snapshots, next.epoch));
  }, [pending]);

  if (!payload) {
    return <section className="playground-panel">Loading…</section>;
  }

  return (
    <section className="playground-panel">
      <DecisionBoundaryChart payload={payload} epoch={epoch} frameIndex={frameIndex} onHover={setHover} />
      <PlaygroundControls
        config={config}
        payload={payload}
        epoch={epoch}
        frameIndex={frameIndex}
        playing={playing}
        hover={hover}
        pending={pending}
        onConfigChange={setConfig}
        onFrameIndexChange={setFrameIndex}
        onPlayingChange={setPlaying}
        onReset={onReset}
        onStep={onStep}
      />
      <p className="playground-hint">Space — play/pause. Hover — inspect probability. Scrubber — replay boundary snapshots.</p>
    </section>
  );
}
