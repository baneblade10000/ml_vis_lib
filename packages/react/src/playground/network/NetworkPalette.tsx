import type { GraphNodeKind } from "@ml-vis/core";
import { PALETTE_DRAG_TYPE } from "./graphAdapter";
import { useNetworkMessages } from "./messages";

function onDragStart(event: React.DragEvent, kind: GraphNodeKind) {
  event.dataTransfer.setData(PALETTE_DRAG_TYPE, kind);
  event.dataTransfer.effectAllowed = "move";
}

const PALETTE_KINDS = ["dense", "sum", "output"] as const;

const PALETTE_ICONS: Record<(typeof PALETTE_KINDS)[number], string> = {
  dense: "D",
  sum: "+",
  output: "O",
};

export function NetworkPalette() {
  const t = useNetworkMessages();
  const labels: Record<(typeof PALETTE_KINDS)[number], { label: string; hint: string }> = {
    dense: { label: t.paletteDense, hint: t.paletteDenseHint },
    sum: { label: t.paletteSum, hint: t.paletteSumHint },
    output: { label: t.paletteOutput, hint: t.paletteOutputHint },
  };

  return (
    <div className="tf-network-palette" data-testid="network-palette">
      <div className="tf-network-palette-title">{t.blocks}</div>
      <div className="tf-network-palette-items">
        {PALETTE_KINDS.map((kind) => (
          <div
            key={kind}
            className="tf-network-palette-item"
            draggable
            data-testid={`network-palette-add-${kind}`}
            onDragStart={(e) => onDragStart(e, kind)}
            title={labels[kind].hint}
          >
            <span className={`tf-network-palette-icon tf-network-palette-icon--${kind}`}>
              {PALETTE_ICONS[kind]}
            </span>
            <span className="tf-network-palette-label">{labels[kind].label}</span>
          </div>
        ))}
      </div>
      <p className="tf-network-palette-hint">{t.paletteHint}</p>
    </div>
  );
}
