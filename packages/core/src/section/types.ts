export type SectionSize = "sm" | "md" | "lg" | "full";

export type SectionDefinition = {
  id: string;
  title: string;
  description?: string;
  size?: SectionSize;
  order?: number;
  tags?: string[];
};

export type SectionState = {
  id: string;
  collapsed: boolean;
  visible: boolean;
};

export type SectionLayoutOptions = {
  columns?: number;
  gap?: number;
  rowHeight?: number;
};

export type SectionRect = {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  columnSpan: number;
  rowSpan: number;
};

export type SectionLayoutResult = {
  rects: SectionRect[];
  totalHeight: number;
  columns: number;
  gap: number;
};

export type SectionStoreSnapshot = {
  activeId: string | null;
  states: Record<string, SectionState>;
  sections: SectionDefinition[];
};
