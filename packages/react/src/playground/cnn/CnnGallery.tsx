import { useLayoutEffect, useRef } from "react";
import { renderValueMatrix } from "@ml-vis/core/charts";
import { ImageExample, SignalExample } from "@ml-vis/core/cnn";
import { useCnnMessages } from "./messages";

export interface CnnGalleryProps {
  mode: "2d" | "1d";
  examples: (ImageExample | SignalExample)[];
  predictions: number[];
  inspectedIndex: number;
  onSelectExample: (index: number) => void;
  datasetId: string;
  onSelectDataset: (id: string) => void;
  datasetIds: string[];
  datasetLabels: Record<string, string>;
}

function ImageThumb({ example, prediction, selected, onClick }: {
  example: ImageExample;
  prediction: number;
  selected: boolean;
  onClick: () => void;
}) {
  const heatRef = useRef<HTMLCanvasElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useLayoutEffect(() => {
    const heat = heatRef.current;
    const canvas = canvasRef.current;
    if (!heat || !canvas) return;
    renderValueMatrix(heat, example.pixels, {
      layout: "row-major",
      palette: "gray",
    });
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, 28, 28);
    ctx.drawImage(heat, 0, 0, 28, 28);
  }, [example.pixels]);

  const predicted = prediction >= 0.5 ? 1 : 0;
  const correct = predicted === example.label;
  return (
    <button
      type="button"
      className={`cnn-gallery-item${selected ? " selected" : ""}${correct ? " correct" : " wrong"}`}
      onClick={onClick}
      title={`label ${example.label} → pred ${predicted.toFixed(2)}`}
    >
      <canvas ref={heatRef} width={example.pixels.length} height={example.pixels.length} hidden aria-hidden />
      <canvas ref={canvasRef} width={28} height={28} className="cnn-gallery-canvas" />
    </button>
  );
}

function SignalThumb({ example, prediction, selected, onClick }: {
  example: SignalExample;
  prediction: number;
  selected: boolean;
  onClick: () => void;
}) {
  const heatRef = useRef<HTMLCanvasElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useLayoutEffect(() => {
    const heat = heatRef.current;
    const canvas = canvasRef.current;
    if (!heat || !canvas) return;
    renderValueMatrix(heat, [example.values], {
      layout: "row-major",
      palette: "gray",
    });
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, 56, 12);
    ctx.drawImage(heat, 0, 0, 56, 12);
  }, [example.values]);

  const predicted = prediction >= 0.5 ? 1 : 0;
  const correct = predicted === example.label;
  return (
    <button
      type="button"
      className={`cnn-gallery-item cnn-gallery-item--signal${selected ? " selected" : ""}${correct ? " correct" : " wrong"}`}
      onClick={onClick}
      title={`label ${example.label} → pred ${predicted.toFixed(2)}`}
    >
      <canvas ref={heatRef} width={example.values.length} height={1} hidden aria-hidden />
      <canvas ref={canvasRef} width={56} height={12} className="cnn-gallery-canvas" />
    </button>
  );
}

/** Input gallery + dataset picker. Items are tinted by prediction correctness. */
export function CnnGallery({
  mode,
  examples,
  predictions,
  inspectedIndex,
  onSelectExample,
  datasetId,
  onSelectDataset,
  datasetIds,
  datasetLabels,
}: CnnGalleryProps) {
  const t = useCnnMessages();
  const shown = examples.slice(0, 24);

  return (
    <div className="cnn-gallery-panel">
      <h4 className="nn-flow-dock-title">{t.dataset}</h4>
      <div className="nn-flat-switch nn-flat-switch--stack" role="group" aria-label={t.dataset}>
        {datasetIds.map((id) => (
          <button
            key={id}
            type="button"
            className={`nn-flat-switch__btn${id === datasetId ? " selected" : ""}`}
            onClick={() => onSelectDataset(id)}
          >
            {datasetLabels[id] ?? id}
          </button>
        ))}
      </div>
      <h4 className="nn-flow-dock-title">{t.gallery}</h4>
      <p className="nn-arch-hint">{t.galleryHint}</p>
      <div className={`cnn-gallery-grid${mode === "1d" ? " cnn-gallery-grid--signal" : ""}`}>
        {shown.map((ex, i) =>
          mode === "2d" ? (
            <ImageThumb
              key={i}
              example={ex as ImageExample}
              prediction={predictions[i] ?? 0.5}
              selected={i === inspectedIndex}
              onClick={() => onSelectExample(i)}
            />
          ) : (
            <SignalThumb
              key={i}
              example={ex as SignalExample}
              prediction={predictions[i] ?? 0.5}
              selected={i === inspectedIndex}
              onClick={() => onSelectExample(i)}
            />
          ),
        )}
      </div>
    </div>
  );
}
