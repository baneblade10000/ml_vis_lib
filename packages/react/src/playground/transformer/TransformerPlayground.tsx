import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TransformerSnapshot, TransformerTaskId } from "@ml-vis/core/transformer";
import { useTransformerMessages } from "./messages";
import { AttentionHeatmap } from "./AttentionHeatmap";
import { TransformerArchitecture } from "./TransformerArchitecture";
import { LossSparkline } from "./LossSparkline";
import { TokenStrip } from "./TokenStrip";

/** Worker protocol mirrored from the playground app's transformerWorker.ts. */
export type ToTransformerWorkerMessage =
  | { type: "init"; task: TransformerTaskId }
  | { type: "play"; stepsPerTick: number }
  | { type: "pause" }
  | { type: "stepOnce" }
  | { type: "reset"; task: TransformerTaskId }
  | { type: "setTask"; task: TransformerTaskId }
  | { type: "setLearningRate"; lr: number }
  | { type: "rerollSample" };

export interface TransformerPlaygroundProps {
  createWorker: () => Worker;
  initialTask?: TransformerTaskId;
  toolbarStart?: React.ReactNode;
  toolbarEnd?: React.ReactNode;
}

export type AttentionKind = "encSelf" | "decSelf" | "cross";

const LEARNING_RATES = [0.01, 0.003, 0.001];

export function TransformerPlayground({
  createWorker,
  initialTask = "translate",
  toolbarStart,
  toolbarEnd,
}: TransformerPlaygroundProps) {
  const t = useTransformerMessages();
  const [snapshot, setSnapshot] = useState<TransformerSnapshot | null>(null);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [task, setTask] = useState<TransformerTaskId>(initialTask);
  const [lr, setLr] = useState(LEARNING_RATES[1]);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState<number | "mean">("mean");
  const [focus, setFocus] = useState<{ kind: AttentionKind; layer: number } | null>(null);
  const workerRef = useRef<Worker | null>(null);

  const send = useCallback((msg: ToTransformerWorkerMessage) => {
    workerRef.current?.postMessage(msg);
  }, []);

  useEffect(() => {
    const worker = createWorker();
    workerRef.current = worker;
    worker.addEventListener("message", (event: MessageEvent) => {
      const data = event.data as
        | { type: "ready"; backend: string }
        | { type: "snapshot"; snapshot: TransformerSnapshot }
        | { type: "error"; message: string };
      if (data.type === "ready") {
        setReady(true);
      } else if (data.type === "snapshot") {
        setSnapshot(data.snapshot);
      } else if (data.type === "error") {
        setError(data.message);
        setPlaying(false);
      }
    });
    worker.addEventListener("error", (event) => {
      // Load failures (server down, 404) carry no message — include the file.
      const detail = event.message || `${event.filename ?? "worker script"}${event.lineno ? `:${event.lineno}` : ""}`;
      setError(detail);
      setPlaying(false);
    });
    send({ type: "init", task: initialTask });
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [createWorker, initialTask, send]);

  // Play/pause is owned by the worker loop; mirror it in local state.
  useEffect(() => {
    if (!ready) return;
    if (playing) send({ type: "play", stepsPerTick: 25 });
    else send({ type: "pause" });
  }, [playing, ready, send]);

  const layers = snapshot ? Math.min(snapshot.attention.encSelf.length, snapshot.attention.decSelf.length) : 0;
  useEffect(() => {
    if (layer >= layers) setLayer(0);
  }, [layers, layer]);

  const onTaskChange = useCallback(
    (next: TransformerTaskId) => {
      setTask(next);
      setPlaying(false);
      send({ type: "setTask", task: next });
    },
    [send],
  );

  const onReset = useCallback(() => {
    setPlaying(false);
    send({ type: "reset", task });
  }, [send, task]);

  const onStep = useCallback(() => {
    setPlaying(false);
    send({ type: "stepOnce" });
  }, [send]);

  const onReroll = useCallback(() => {
    setPlaying(false);
    send({ type: "rerollSample" });
  }, [send]);

  const onLearningRate = useCallback(
    (next: number) => {
      setLr(next);
      send({ type: "setLearningRate", lr: next });
    },
    [send],
  );

  const stats = useMemo(() => {
    if (!snapshot) return null;
    return {
      step: snapshot.step,
      loss: snapshot.loss,
      accuracy: snapshot.accuracy,
    };
  }, [snapshot]);

  const labels = snapshot?.labels ?? [];
  const encLabels = snapshot?.encInTokens.map((tok) => labels[tok] ?? "?") ?? [];
  const decLabels = snapshot?.decInTokens.map((tok) => labels[tok] ?? "?") ?? [];

  return (
    <div className="tf-playground">
      <div className="nn-immersive-toolbar">
        <div className="nn-toolbar-group nn-toolbar-group--actions">
          {toolbarStart}
          <button type="button" className="nn-btn nn-btn--ghost" onClick={onReset}>
            {t.reset}
          </button>
          <button
            type="button"
            className={`nn-btn nn-btn--primary${playing ? " playing" : ""}`}
            onClick={() => setPlaying((p) => !p)}
            disabled={!ready}
          >
            {playing ? t.pause : t.play}
          </button>
          <button type="button" className="nn-btn nn-btn--secondary" onClick={onStep} disabled={!ready}>
            {t.step}
          </button>
          <button type="button" className="nn-btn nn-btn--ghost" onClick={onReroll} disabled={!ready}>
            ↻
          </button>
          <div className="nn-flat-switch" role="group" aria-label={t.task}>
            {(["translate", "reverse"] as const).map((id) => (
              <button
                key={id}
                type="button"
                className={`nn-flat-switch__btn${task === id ? " selected" : ""}`}
                title={id === "translate" ? t.taskTranslateHint : t.taskReverseHint}
                onClick={() => onTaskChange(id)}
              >
                {id === "translate" ? t.taskTranslate : t.taskReverse}
              </button>
            ))}
          </div>
          <label className="tf-toolbar-lr">
            <span>{t.learningRate}</span>
            <select value={lr} onChange={(e) => onLearningRate(Number(e.target.value))}>
              {LEARNING_RATES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="nn-inspired-by">
          {t.inspiredBy}{" "}
          <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noopener noreferrer">
            {t.inspiredBySource}
          </a>
        </p>
        <div className="nn-toolbar-group nn-toolbar-group--params">
          <div className="nn-toolbar-stat">
            <span className="label">{t.stepCount}</span>
            <span className="value">{stats ? stats.step.toLocaleString() : "—"}</span>
          </div>
          <div className="nn-toolbar-stat">
            <span className="label">{t.loss}</span>
            <span className="value">{stats ? stats.loss.toFixed(3) : "—"}</span>
          </div>
          <div className="nn-toolbar-stat nn-toolbar-stat--train">
            <span className="label">{t.accuracy}</span>
            <span className="value">
              {stats ? `${Math.round(stats.accuracy * 100)}%` : "—"}
            </span>
          </div>
          {toolbarEnd}
        </div>
      </div>

      <div className="tf-body">
        {error && <div className="tf-error">{t.wasmError}: {error}</div>}
        {!snapshot ? (
          <div className="tf-loading">{t.loadingWasm}</div>
        ) : (
          <>
            <div className="tf-tokens-row">
              <div className="tf-token-group">
                <span className="tf-token-group__label">{t.inputSequence}</span>
                <TokenStrip tokens={snapshot.inputTokens} labels={snapshot.labels} />
              </div>
              <span className="tf-transform-badge" title={task === "translate" ? t.taskTranslateHint : t.taskReverseHint}>
                {task === "translate" ? "RU → EN" : "⟲"}
              </span>
              <div className="tf-token-group">
                <span className="tf-token-group__label">{t.targetSequence}</span>
                <TokenStrip tokens={snapshot.targetTokens} labels={snapshot.labels} tone="target" />
              </div>
              <div className="tf-token-group">
                <span className="tf-token-group__label">{t.prediction}</span>
                <TokenStrip
                  tokens={snapshot.predictedTokens}
                  labels={snapshot.labels}
                  tone="prediction"
                  expected={snapshot.targetTokens}
                />
              </div>
              <div className="tf-loss-card">
                <span className="tf-token-group__label">{t.lossCurve}</span>
                <LossSparkline history={snapshot.lossHistory} />
              </div>
            </div>

            <div className="tf-main">
              <TransformerArchitecture
                snapshot={snapshot}
                focus={focus}
                onFocusChange={(next) => {
                  setFocus(next);
                  if (next) setLayer(next.layer);
                }}
              />
              <div className="tf-attention-panel">
                <div className="tf-attention-controls">
                  <div className="tf-attention-controls__group">
                    <span>{t.layer}</span>
                    <div className="nn-flat-switch">
                      {Array.from({ length: layers }, (_, i) => (
                        <button
                          key={i}
                          type="button"
                          className={`nn-flat-switch__btn${layer === i ? " selected" : ""}`}
                          onClick={() => setLayer(i)}
                        >
                          {i + 1}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="tf-attention-controls__group">
                    <span>{t.head}</span>
                    <div className="nn-flat-switch">
                      <button
                        type="button"
                        className={`nn-flat-switch__btn${head === "mean" ? " selected" : ""}`}
                        onClick={() => setHead("mean")}
                      >
                        {t.headMean}
                      </button>
                      {(snapshot.attention.encSelf[0] ?? []).map((_, h) => (
                        <button
                          key={h}
                          type="button"
                          className={`nn-flat-switch__btn${head === h ? " selected" : ""}`}
                          onClick={() => setHead(h)}
                        >
                          H{h + 1}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="tf-heatmaps">
                  <AttentionHeatmap
                    title={t.encSelfTitle}
                    hint={t.encSelfHint}
                    matrices={snapshot.attention.encSelf[layer]}
                    head={head}
                    rowLabels={encLabels}
                    colLabels={encLabels}
                    rowTokens={snapshot.encInTokens}
                    colTokens={snapshot.encInTokens}
                    alphabetSize={snapshot.alphabetSize}
                  />
                  <AttentionHeatmap
                    title={t.decSelfTitle}
                    hint={t.decSelfHint}
                    matrices={snapshot.attention.decSelf[layer]}
                    head={head}
                    rowLabels={decLabels}
                    colLabels={decLabels}
                    rowTokens={snapshot.decInTokens}
                    colTokens={snapshot.decInTokens}
                    alphabetSize={snapshot.alphabetSize}
                  />
                  <AttentionHeatmap
                    title={t.crossTitle}
                    hint={t.crossHint}
                    matrices={snapshot.attention.cross[layer]}
                    head={head}
                    rowLabels={decLabels}
                    colLabels={encLabels}
                    rowTokens={snapshot.decInTokens}
                    colTokens={snapshot.encInTokens}
                    alphabetSize={snapshot.alphabetSize}
                  />
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
