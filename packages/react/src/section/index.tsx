import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import {
  SectionRegistry,
  SectionStore,
  type SectionDefinition,
  type SectionLayoutOptions,
  type SectionStoreSnapshot,
} from "@ml-vis/core";
import { useI18n } from "../i18n";

type SectionContextValue = {
  store: SectionStore;
  snapshot: SectionStoreSnapshot;
};

const SectionContext = createContext<SectionContextValue | null>(null);

export type SectionProviderProps = {
  sections: SectionDefinition[];
  children: ReactNode;
  defaultActiveId?: string;
};

export function SectionProvider({
  sections,
  children,
  defaultActiveId,
}: SectionProviderProps) {
  const storeRef = useRef<SectionStore | null>(null);
  if (!storeRef.current) {
    const registry = new SectionRegistry();
    storeRef.current = new SectionStore(registry, sections);
    if (defaultActiveId) {
      storeRef.current.setActive(defaultActiveId);
    }
  }

  const store = storeRef.current;
  const [snapshot, setSnapshot] = useState(() => store.getSnapshot());

  useEffect(() => {
    return store.subscribe(() => setSnapshot(store.getSnapshot()));
  }, [store]);

  useEffect(() => {
    const currentIds = new Set(store.listSections().map((s) => s.id));
    const nextIds = new Set(sections.map((s) => s.id));

    for (const section of sections) {
      if (!currentIds.has(section.id)) {
        store.addSection(section);
      }
    }

    for (const id of currentIds) {
      if (!nextIds.has(id)) {
        store.removeSection(id);
      }
    }
  }, [sections, store]);

  const value = useMemo(() => ({ store, snapshot }), [store, snapshot]);

  return (
    <SectionContext.Provider value={value}>{children}</SectionContext.Provider>
  );
}

export function useSections() {
  const ctx = useContext(SectionContext);
  if (!ctx) {
    throw new Error("useSections must be used within SectionProvider");
  }
  return ctx;
}

export type SectionLayoutProps = SectionLayoutOptions & {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
};

export function SectionLayout({
  children,
  columns = 2,
  gap = 16,
  rowHeight = 280,
  className,
  style,
}: SectionLayoutProps) {
  return (
    <div
      className={className}
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
        gap,
        alignItems: "start",
        ...style,
      }}
      data-row-height={rowHeight}
    >
      {children}
    </div>
  );
}

const SIZE_STYLES: Record<string, CSSProperties> = {
  sm: { gridColumn: "span 1" },
  md: { gridColumn: "span 1" },
  lg: { gridColumn: "span 2" },
  full: { gridColumn: "1 / -1" },
};

export type SectionProps = {
  id: string;
  title: string;
  description?: string;
  size?: SectionDefinition["size"];
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
};

export function Section({
  id,
  title,
  description,
  size = "md",
  children,
  className,
  style,
}: SectionProps) {
  const { store, snapshot } = useSections();
  const { t } = useI18n();
  const state = snapshot.states[id] ?? { id, collapsed: false, visible: true };
  const isActive = snapshot.activeId === id;

  if (!state.visible) return null;

  return (
    <section
      id={`section-${id}`}
      className={className}
      data-active={isActive}
      data-collapsed={state.collapsed}
      style={{
        background: "#fff",
        borderRadius: 12,
        border: isActive ? "2px solid #3b82f6" : "1px solid #e2e8f0",
        boxShadow: "0 1px 3px rgb(0 0 0 / 0.06)",
        overflow: "hidden",
        ...SIZE_STYLES[size],
        ...style,
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          gap: "0.75rem",
          padding: "1rem 1.25rem",
          borderBottom: state.collapsed ? "none" : "1px solid #f1f5f9",
          cursor: "pointer",
          userSelect: "none",
        }}
        onClick={() => store.setActive(id)}
      >
        <div>
          <h2 style={{ margin: 0, fontSize: "1rem", fontWeight: 600, color: "#0f172a" }}>
            {title}
          </h2>
          {description && !state.collapsed && (
            <p style={{ margin: "0.25rem 0 0", fontSize: "0.8125rem", color: "#64748b" }}>
              {description}
            </p>
          )}
        </div>
        <button
          type="button"
          aria-label={state.collapsed ? t("expandSection") : t("collapseSection")}
          onClick={(e) => {
            e.stopPropagation();
            store.toggleCollapsed(id);
          }}
          style={{
            border: "1px solid #e2e8f0",
            background: "#f8fafc",
            borderRadius: 6,
            width: 28,
            height: 28,
            cursor: "pointer",
            flexShrink: 0,
            fontSize: "0.875rem",
            color: "#475569",
          }}
        >
          {state.collapsed ? "+" : "−"}
        </button>
      </header>
      {!state.collapsed && (
        <div style={{ padding: "1.25rem" }}>{children}</div>
      )}
    </section>
  );
}

export type SectionNavProps = {
  className?: string;
  style?: CSSProperties;
};

export function SectionNav({ className, style }: SectionNavProps) {
  const { store, snapshot } = useSections();
  const { t } = useI18n();

  return (
    <nav
      className={className}
      aria-label={t("sectionsNav")}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "0.25rem",
        ...style,
      }}
    >
      {snapshot.sections.map((section) => {
        const state = snapshot.states[section.id];
        if (state && !state.visible) return null;
        const isActive = snapshot.activeId === section.id;

        return (
          <button
            key={section.id}
            type="button"
            onClick={() => {
              store.setActive(section.id);
              document.getElementById(`section-${section.id}`)?.scrollIntoView({
                behavior: "smooth",
                block: "start",
              });
            }}
            style={{
              textAlign: "left",
              padding: "0.5rem 0.75rem",
              borderRadius: 8,
              border: "none",
              background: isActive ? "#eff6ff" : "transparent",
              color: isActive ? "#1d4ed8" : "#475569",
              fontWeight: isActive ? 600 : 400,
              fontSize: "0.875rem",
              cursor: "pointer",
            }}
          >
            {section.title}
          </button>
        );
      })}
    </nav>
  );
}
