import type { GraphNodeKind } from "@ml-vis/core";
import { PALETTE_DRAG_TYPE } from "./graphAdapter";
import { useNetworkMessages } from "./messages";

/** MLP-only: a single draggable dense (neuron) block. */
const PALETTE_KINDS = ["dense"] as const;

function onDragStart(event: React.DragEvent, kind: GraphNodeKind) {
  event.dataTransfer.setData(PALETTE_DRAG_TYPE, kind);
  event.dataTransfer.setData("text/plain", kind);
  event.dataTransfer.effectAllowed = "move";
}

export function NetworkPalette() {
  const t = useNetworkMessages();

  return (
    <div className="nn-network-palette" data-testid="network-palette">
      <div className="nn-network-palette-title">{t.blocks}</div>
      <div className="nn-network-palette-items">
        {PALETTE_KINDS.map((kind) => (
          <div
            key={kind}
            className="nn-network-palette-item"
            draggable
            data-testid={`network-palette-add-${kind}`}
            onDragStart={(e) => onDragStart(e, kind)}
            title={t.paletteDenseHint}
          >
            <span className={`nn-network-palette-icon nn-network-palette-icon--${kind}`}>D</span>
            <span className="nn-network-palette-label">{t.paletteDense}</span>
          </div>
        ))}
      </div>
      <p className="nn-network-palette-hint">{t.paletteHint}</p>
    </div>
  );
}
