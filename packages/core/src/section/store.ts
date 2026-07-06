import { computeSectionLayout } from "./layout";
import { SectionRegistry } from "./registry";
import type {
  SectionDefinition,
  SectionLayoutOptions,
  SectionLayoutResult,
  SectionState,
  SectionStoreSnapshot,
} from "./types";

type Listener = () => void;

function defaultState(id: string): SectionState {
  return { id, collapsed: false, visible: true };
}

export class SectionStore {
  private registry: SectionRegistry;
  private states = new Map<string, SectionState>();
  private activeId: string | null = null;
  private listeners = new Set<Listener>();

  constructor(registry: SectionRegistry, sections?: SectionDefinition[]) {
    this.registry = registry;
    if (sections) {
      this.registry.registerMany(sections);
      for (const section of sections) {
        this.states.set(section.id, defaultState(section.id));
      }
      if (sections.length > 0) {
        this.activeId = sortByOrder(sections)[0]!.id;
      }
    }
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getSnapshot(): SectionStoreSnapshot {
    return {
      activeId: this.activeId,
      states: Object.fromEntries(this.states),
      sections: this.registry.list(),
    };
  }

  listSections(): SectionDefinition[] {
    return this.registry.list();
  }

  getState(id: string): SectionState {
    return this.states.get(id) ?? defaultState(id);
  }

  setActive(id: string): void {
    if (!this.registry.has(id)) return;
    this.activeId = id;
    this.emit();
  }

  getActive(): string | null {
    return this.activeId;
  }

  setCollapsed(id: string, collapsed: boolean): void {
    if (!this.registry.has(id)) return;
    const state = this.getState(id);
    this.states.set(id, { ...state, collapsed });
    this.emit();
  }

  toggleCollapsed(id: string): void {
    const state = this.getState(id);
    this.setCollapsed(id, !state.collapsed);
  }

  setVisible(id: string, visible: boolean): void {
    if (!this.registry.has(id)) return;
    const state = this.getState(id);
    this.states.set(id, { ...state, visible });
    this.emit();
  }

  addSection(section: SectionDefinition): void {
    this.registry.register(section);
    if (!this.states.has(section.id)) {
      this.states.set(section.id, defaultState(section.id));
    }
    if (!this.activeId) {
      this.activeId = section.id;
    }
    this.emit();
  }

  removeSection(id: string): void {
    this.registry.unregister(id);
    this.states.delete(id);
    if (this.activeId === id) {
      const remaining = this.registry.list();
      this.activeId = remaining[0]?.id ?? null;
    }
    this.emit();
  }

  computeLayout(
    options: SectionLayoutOptions & { containerWidth: number },
  ): SectionLayoutResult {
    return computeSectionLayout(
      this.registry.list(),
      options,
      Object.fromEntries(this.states),
    );
  }

  private emit(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }
}

function sortByOrder(sections: SectionDefinition[]): SectionDefinition[] {
  return [...sections].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
}
