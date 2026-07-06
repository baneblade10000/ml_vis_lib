import { sortSections } from "./layout";
import type { SectionDefinition } from "./types";

export class SectionRegistry {
  private sections = new Map<string, SectionDefinition>();

  register(section: SectionDefinition): void {
    this.sections.set(section.id, { ...section });
  }

  registerMany(sections: SectionDefinition[]): void {
    for (const section of sections) {
      this.register(section);
    }
  }

  update(id: string, patch: Partial<SectionDefinition>): void {
    const existing = this.sections.get(id);
    if (!existing) return;
    this.sections.set(id, { ...existing, ...patch, id });
  }

  unregister(id: string): boolean {
    return this.sections.delete(id);
  }

  get(id: string): SectionDefinition | undefined {
    const section = this.sections.get(id);
    return section ? { ...section } : undefined;
  }

  has(id: string): boolean {
    return this.sections.has(id);
  }

  list(): SectionDefinition[] {
    return sortSections([...this.sections.values()]);
  }

  clear(): void {
    this.sections.clear();
  }
}
