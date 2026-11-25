import { playgroundMessages } from "../i18n";
import { visualizations, type VisualizationId } from "../visualizations";
import { catalogKnowledgeTree, type CatalogTreeNode } from "./tree";

export interface CatalogSearch {
  /** False for a blank query: nothing is dimmed or filtered. */
  searching: boolean;
  /** Tree nodes whose label contains the query; the root never counts. */
  matchedNodes: Set<string>;
  /** Visualizations whose title or description contains the query. */
  matchedViz: Set<VisualizationId>;
  /**
   * Node and visualization ids that stay visible: every hit, the path from it
   * up to the root, and the whole branch under a hit node. The graph dims the
   * rest, the list drops it.
   */
  lit: Set<string>;
}

/** Case-insensitive substring search shared by the graph and the list views. */
export function searchCatalog(locale: "en" | "ru", query: string): CatalogSearch {
  const messages = playgroundMessages[locale];
  const needle = query.trim().toLowerCase();
  const search: CatalogSearch = {
    searching: needle.length > 0,
    matchedNodes: new Set(),
    matchedViz: new Set(),
    lit: new Set(),
  };
  if (!search.searching) return search;

  const matches = (text: string) => text.toLowerCase().includes(needle);
  for (const viz of visualizations) {
    if (matches(messages[viz.titleKey]) || matches(messages[viz.descriptionKey])) {
      search.matchedViz.add(viz.id);
    }
  }

  // Depth-first: a node is lit when it is a hit, sits inside a hit branch, or
  // leads to one. Its cards are lit with it.
  const walk = (node: CatalogTreeNode, insideHit: boolean): boolean => {
    const hit = node !== catalogKnowledgeTree && matches(locale === "ru" ? node.label.ru : node.label.en);
    if (hit) search.matchedNodes.add(node.id);

    const branchHit = insideHit || hit;
    let hitBelow = false;
    for (const child of node.children ?? []) {
      if (walk(child, branchHit)) hitBelow = true;
    }
    const cardHit = node.vizIds?.some((vizId) => search.matchedViz.has(vizId));
    if (!(branchHit || hitBelow || cardHit)) return false;

    search.lit.add(node.id);
    for (const vizId of node.vizIds ?? []) search.lit.add(vizId);
    return true;
  };
  walk(catalogKnowledgeTree, false);

  // A visualization the tree doesn't reference is still shown when it matches.
  for (const vizId of search.matchedViz) search.lit.add(vizId);
  return search;
}
