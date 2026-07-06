import { describe, expect, it } from "vitest";
import { downsample, extent } from "./utils/math";
import { computeSectionLayout, SectionRegistry, SectionStore } from "./section";

describe("extent", () => {
  it("returns min and max of numeric array", () => {
    expect(extent([3, 1, 4, 1, 5])).toEqual([1, 5]);
  });

  it("returns null for empty array", () => {
    expect(extent([])).toBeNull();
  });
});

describe("downsample", () => {
  it("returns all points when under limit", () => {
    const points = [
      { x: 1, y: 2 },
      { x: 3, y: 4 },
    ];
    expect(downsample(points, 10)).toEqual(points);
  });

  it("reduces points to target count", () => {
    const points = Array.from({ length: 100 }, (_, i) => ({ x: i, y: i }));
    const result = downsample(points, 10);
    expect(result).toHaveLength(10);
  });
});

describe("computeSectionLayout", () => {
  const sections = [
    { id: "a", title: "A", size: "lg" as const, order: 0 },
    { id: "b", title: "B", size: "md" as const, order: 1 },
    { id: "c", title: "C", size: "sm" as const, order: 2 },
  ];

  it("places sections in a grid", () => {
    const layout = computeSectionLayout(sections, { containerWidth: 800, columns: 2 });
    expect(layout.rects).toHaveLength(3);
    expect(layout.totalHeight).toBeGreaterThan(0);
  });

  it("skips hidden sections", () => {
    const layout = computeSectionLayout(sections, { containerWidth: 800, columns: 2 }, {
      b: { id: "b", collapsed: false, visible: false },
    });
    expect(layout.rects.find((r) => r.id === "b")).toBeUndefined();
  });

  it("reduces height for collapsed sections", () => {
    const expanded = computeSectionLayout(sections, { containerWidth: 800, columns: 2 });
    const collapsed = computeSectionLayout(sections, { containerWidth: 800, columns: 2 }, {
      a: { id: "a", collapsed: true, visible: true },
    });
    expect(collapsed.totalHeight).toBeLessThan(expanded.totalHeight);
  });
});

describe("SectionStore", () => {
  it("manages active section and collapse state", () => {
    const registry = new SectionRegistry();
    const store = new SectionStore(registry, [
      { id: "metrics", title: "Metrics", order: 0 },
      { id: "embeddings", title: "Embeddings", order: 1 },
    ]);

    expect(store.getActive()).toBe("metrics");

    store.setActive("embeddings");
    expect(store.getActive()).toBe("embeddings");

    store.toggleCollapsed("metrics");
    expect(store.getState("metrics").collapsed).toBe(true);

    const layout = store.computeLayout({ containerWidth: 960, columns: 2 });
    expect(layout.rects.length).toBeGreaterThan(0);
  });
});
