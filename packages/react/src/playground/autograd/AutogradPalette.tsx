import { AUTOGRAD_PALETTE_OPS, OP_SPECS } from "@ml-vis/core";
import { AUTOGRAD_DRAG_TYPE } from "./adapter";
import { useAutogradMessages } from "./messages";

function onDragStart(event: React.DragEvent, op: string) {
  event.dataTransfer.setData(AUTOGRAD_DRAG_TYPE, op);
  event.dataTransfer.effectAllowed = "move";
}

export function AutogradPalette() {
  const t = useAutogradMessages();

  return (
    <div className="tf-network-palette" data-testid="autograd-palette">
      <div className="tf-network-palette-title">{t.blocks}</div>
      <div className="tf-network-palette-items ag-palette-items">
        {AUTOGRAD_PALETTE_OPS.map((op) => (
          <div
            key={op}
            className="tf-network-palette-item ag-palette-item"
            draggable
            data-testid={`autograd-palette-add-${op}`}
            onDragStart={(e) => onDragStart(e, op)}
            title={t.opLabels[op]}
          >
            <span className="tf-network-palette-icon ag-palette-icon">{OP_SPECS[op].symbol}</span>
            <span className="tf-network-palette-label">{t.opLabels[op]}</span>
          </div>
        ))}
      </div>
      <p className="tf-network-palette-hint">{t.paletteHint}</p>
    </div>
  );
}
