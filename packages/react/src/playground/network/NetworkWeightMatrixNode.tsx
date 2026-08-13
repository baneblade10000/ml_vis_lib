import { useContext, useState, createContext, type ReactNode } from "react";
import { type Node, type NodeProps } from "@xyflow/react";
import { weightColor, weightValueNormalized } from "@ml-vis/core/network";
import type { NetworkNodeData, WeightMatrixCellData } from "./graphAdapter";

export type NetworkGraphActions = {
  onSelectNode: (nodeId: string | null) => void;
  onSelectEdge: (edgeId: string | null) => void;
};

export const NetworkGraphActionsContext = createContext<NetworkGraphActions>({
  onSelectNode: () => {},
  onSelectEdge: () => {},
});

export function NetworkGraphActionsProvider({
  value,
  children,
}: {
  value: NetworkGraphActions;
  children: ReactNode;
}) {
  return (
    <NetworkGraphActionsContext.Provider value={value}>{children}</NetworkGraphActionsContext.Provider>
  );
}

function cellVizValue(
  cell: WeightMatrixCellData,
  vizMode: "weight" | "gradient",
  gradScale: number,
): number {
  if (vizMode === "gradient") {
    const ratio = Math.abs(cell.gradient) / gradScale;
    return Math.sign(cell.gradient) * Math.sqrt(ratio);
  }
  return weightValueNormalized(cell.weight);
}

/**
 * Interactive weight matrix between two layer columns. Rows = destination
 * neurons, columns = sources. Hover lights the row/column; click selects the
 * underlying link (same inspector as graph edges).
 */
export function WeightMatrixFlowNode({ data }: NodeProps<Node<NetworkNodeData>>) {
  const matrix = data.matrix;
  const { onSelectEdge, onSelectNode } = useContext(NetworkGraphActionsContext);
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null);

  if (!matrix) return null;

  const { cellPx, cells, selectedEdgeId, selectedNodeId, vizMode, learningRate, gradScale } = matrix;
  const cols = matrix.sourceIds.length;
  const rows = matrix.destIds.length;
  const selectedCol = selectedNodeId ? matrix.sourceIds.indexOf(selectedNodeId) : -1;
  const selectedRow = selectedNodeId ? matrix.destIds.indexOf(selectedNodeId) : -1;
  const nodeInMatrix = selectedCol >= 0 || selectedRow >= 0;
  const hoverCell =
    hover && cells[hover.row]?.[hover.col]?.linkId ? cells[hover.row]![hover.col]! : null;

  const signed = (n: number, places: number) => `${n >= 0 ? "+" : ""}${n.toFixed(places)}`;

  return (
    <div
      className="nn-weight-matrix nodrag nopan"
      style={{ width: cols * cellPx + 2, height: rows * cellPx + 2 }}
      onMouseLeave={() => setHover(null)}
    >
      <div
        className="nn-weight-matrix__grid"
        style={{
          gridTemplateColumns: `repeat(${cols}, ${cellPx}px)`,
          gridTemplateRows: `repeat(${rows}, ${cellPx}px)`,
        }}
      >
        {cells.map((row, r) =>
          row.map((cell, c) => {
            const empty = !cell.linkId;
            const selected = !!cell.linkId && cell.linkId === selectedEdgeId;
            const rowHi = hover?.row === r || r === selectedRow;
            const colHi = hover?.col === c || c === selectedCol;
            const band = !empty && (r === selectedRow || c === selectedCol);
            const cross = rowHi || colHi;
            const dim = nodeInMatrix && !empty && !band && hover?.row !== r && hover?.col !== c;
            const value = empty ? 0 : cellVizValue(cell, vizMode, gradScale);
            const fill = empty
              ? "transparent"
              : cell.active
                ? weightColor(value)
                : "rgba(148,163,184,0.2)";
            return (
              <button
                key={`${r}-${c}`}
                type="button"
                className={[
                  "nn-weight-matrix__cell",
                  empty ? "nn-weight-matrix__cell--empty" : "",
                  selected ? "nn-weight-matrix__cell--selected" : "",
                  band ? "nn-weight-matrix__cell--band" : "",
                  cross && !empty ? "nn-weight-matrix__cell--cross" : "",
                  dim ? "nn-weight-matrix__cell--dim" : "",
                  hover?.row === r && hover?.col === c && !empty
                    ? "nn-weight-matrix__cell--hover"
                    : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                style={{ background: fill, width: cellPx, height: cellPx }}
                disabled={empty}
                aria-label={
                  empty
                    ? undefined
                    : `${matrix.sourceLabels[c]} → ${matrix.destLabels[r]}: ${cell.weight.toFixed(3)}`
                }
                onMouseEnter={() => setHover({ row: r, col: c })}
                onFocus={() => setHover({ row: r, col: c })}
                onClick={(e) => {
                  e.stopPropagation();
                  if (!cell.linkId) return;
                  onSelectNode(null);
                  onSelectEdge(cell.linkId);
                }}
              />
            );
          }),
        )}
      </div>
      {hoverCell && hover && (
        <div className="nn-weight-matrix__tooltip" role="tooltip">
          <span className="nn-weight-matrix__tooltip-pair">
            {matrix.sourceLabels[hover.col]} → {matrix.destLabels[hover.row]}
          </span>
          {vizMode === "gradient" ? (
            <>
              <span>∂ {signed(hoverCell.gradient, 3)}</span>
              <span className="nn-weight-matrix__tooltip-delta">
                → Δw {signed(-(learningRate ?? 0) * hoverCell.gradient, 3)}
              </span>
            </>
          ) : (
            <span>{signed(hoverCell.weight, 2)}</span>
          )}
        </div>
      )}
    </div>
  );
}
