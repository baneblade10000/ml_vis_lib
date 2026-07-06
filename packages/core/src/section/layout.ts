import type {
  SectionDefinition,
  SectionLayoutOptions,
  SectionLayoutResult,
  SectionRect,
  SectionSize,
  SectionState,
} from "./types";

const SIZE_SPANS: Record<SectionSize, { columnSpan: number; rowSpan: number }> = {
  sm: { columnSpan: 1, rowSpan: 1 },
  md: { columnSpan: 1, rowSpan: 2 },
  lg: { columnSpan: 2, rowSpan: 2 },
  full: { columnSpan: 0, rowSpan: 2 },
};

const DEFAULT_GAP = 16;
const DEFAULT_ROW_HEIGHT = 280;

function resolveSpans(size: SectionSize, columns: number) {
  const spans = SIZE_SPANS[size];
  return {
    columnSpan: size === "full" ? columns : Math.min(spans.columnSpan, columns),
    rowSpan: spans.rowSpan,
  };
}

type GridCell = { row: number; col: number };

function findPlacement(
  occupied: boolean[][],
  columnSpan: number,
  rowSpan: number,
  columns: number,
): GridCell | null {
  const maxRows = occupied.length;
  for (let row = 0; row < maxRows + 1; row++) {
    for (let col = 0; col <= columns - columnSpan; col++) {
      if (canPlace(occupied, row, col, columnSpan, rowSpan, columns)) {
        return { row, col };
      }
    }
  }
  return null;
}

function canPlace(
  occupied: boolean[][],
  row: number,
  col: number,
  columnSpan: number,
  rowSpan: number,
  columns: number,
): boolean {
  for (let r = row; r < row + rowSpan; r++) {
    for (let c = col; c < col + columnSpan; c++) {
      if (c >= columns) return false;
      if (occupied[r]?.[c]) return false;
    }
  }
  return true;
}

function markOccupied(
  occupied: boolean[][],
  row: number,
  col: number,
  columnSpan: number,
  rowSpan: number,
): void {
  for (let r = row; r < row + rowSpan; r++) {
    if (!occupied[r]) occupied[r] = Array.from({ length: 10 }, () => false);
    for (let c = col; c < col + columnSpan; c++) {
      occupied[r]![c] = true;
    }
  }
}

export function sortSections(sections: SectionDefinition[]): SectionDefinition[] {
  return [...sections].sort((a, b) => (a.order ?? 0) - (b.order ?? 0) || a.id.localeCompare(b.id));
}

export function computeSectionLayout(
  sections: SectionDefinition[],
  options: SectionLayoutOptions & { containerWidth: number },
  states?: Record<string, SectionState>,
): SectionLayoutResult {
  const columns = Math.max(1, options.columns ?? 2);
  const gap = options.gap ?? DEFAULT_GAP;
  const rowHeight = options.rowHeight ?? DEFAULT_ROW_HEIGHT;
  const columnWidth = (options.containerWidth - gap * (columns - 1)) / columns;

  const visible = sortSections(sections).filter((s) => states?.[s.id]?.visible !== false);
  const occupied: boolean[][] = [];
  const rects: SectionRect[] = [];
  let maxRow = 0;

  for (const section of visible) {
    const collapsed = states?.[section.id]?.collapsed ?? false;
    const size = section.size ?? "md";
    const { columnSpan, rowSpan } = resolveSpans(size, columns);
    const effectiveRowSpan = collapsed ? 1 : rowSpan;

    const placement = findPlacement(occupied, columnSpan, effectiveRowSpan, columns);
    if (!placement) continue;

    const { row, col } = placement;
    markOccupied(occupied, row, col, columnSpan, effectiveRowSpan);

    const width = columnWidth * columnSpan + gap * (columnSpan - 1);
    const height =
      rowHeight * effectiveRowSpan + gap * (effectiveRowSpan - 1);

    rects.push({
      id: section.id,
      x: col * (columnWidth + gap),
      y: row * (rowHeight + gap),
      width,
      height,
      columnSpan,
      rowSpan: effectiveRowSpan,
    });

    maxRow = Math.max(maxRow, row + effectiveRowSpan);
  }

  const totalHeight = maxRow > 0 ? maxRow * (rowHeight + gap) - gap : 0;

  return { rects, totalHeight, columns, gap };
}
