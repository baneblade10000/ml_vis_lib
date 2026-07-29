import { AUTOGRAD_PALETTE_OPS, OP_SPECS, type AutogradOp } from "@ml-vis/core";
import { AUTOGRAD_DRAG_TYPE } from "./adapter";
import { useAutogradMessages } from "./messages";

function onDragStart(event: React.DragEvent, op: AutogradOp) {
  event.dataTransfer.setData(AUTOGRAD_DRAG_TYPE, op);
  event.dataTransfer.setData("text/plain", op);
  event.dataTransfer.effectAllowed = "move";
}

export function AutogradPalette({ onAddOp }: { onAddOp: (op: AutogradOp) => void }) {
  const t = useAutogradMessages();

  return (
    <div className="tf-network-palette ag-palette" data-testid="autograd-palette">
      <div className="tf-network-palette-title">{t.blocks}</div>
      <div className="tf-network-palette-items ag-palette-items">
        {AUTOGRAD_PALETTE_OPS.map((op) => (
          <div
            key={op}
            className="tf-network-palette-item ag-palette-item"
            draggable
            role="button"
            tabIndex={0}
            data-testid={`autograd-palette-add-${op}`}
            onDragStart={(e) => onDragStart(e, op)}
            onClick={() => onAddOp(op)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onAddOp(op);
              }
            }}
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
