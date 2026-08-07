import { useCnnMessages } from "./messages";
import type { CnnDragKind } from "./cnnAdapter";
import { PALETTE_DRAG_TYPE } from "./cnnAdapter";

interface PaletteItem {
  kind: CnnDragKind;
  icon: string;
  labelKey: "paletteConv" | "palettePool" | "paletteDense";
  hintKey: "paletteConvHint" | "palettePoolHint" | "paletteDenseHint";
}

const PALETTE: PaletteItem[] = [
  { kind: "conv", icon: "⊞", labelKey: "paletteConv", hintKey: "paletteConvHint" },
  { kind: "pool", icon: "⊓", labelKey: "palettePool", hintKey: "palettePoolHint" },
  { kind: "dense", icon: "▦", labelKey: "paletteDense", hintKey: "paletteDenseHint" },
];

function onDragStart(event: React.DragEvent, kind: CnnDragKind) {
  event.dataTransfer.setData(PALETTE_DRAG_TYPE, kind);
  event.dataTransfer.setData("text/plain", kind);
  event.dataTransfer.effectAllowed = "copy";
}

export interface CnnPaletteProps {
  onAddLayer: (kind: CnnDragKind) => void;
}

/** Clickable / draggable layer blocks that append to the pipeline. */
export function CnnPalette({ onAddLayer }: CnnPaletteProps) {
  const t = useCnnMessages();

  return (
    <div className="cnn-palette" data-testid="cnn-palette">
      <div className="tf-flow-dock-title">{t.blocks}</div>
      <div className="cnn-palette-items">
        {PALETTE.map((item) => (
          <button
            key={item.kind}
            type="button"
            className="cnn-palette-item"
            draggable
            onDragStart={(e) => onDragStart(e, item.kind)}
            onClick={() => onAddLayer(item.kind)}
            title={t[item.hintKey]}
          >
            <span className={`cnn-palette-icon cnn-palette-icon--${item.kind}`}>{item.icon}</span>
            <span className="cnn-palette-label">{t[item.labelKey]}</span>
          </button>
        ))}
      </div>
      <p className="cnn-palette-hint">{t.paletteHint}</p>
    </div>
  );
}
