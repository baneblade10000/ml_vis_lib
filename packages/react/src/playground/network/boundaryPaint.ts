type BoundaryPainter = () => void;

const painters = new Map<string, Set<BoundaryPainter>>();

export function registerBoundaryPainter(nodeId: string, paint: BoundaryPainter): () => void {
  let set = painters.get(nodeId);
  if (!set) {
    set = new Set();
    painters.set(nodeId, set);
  }
  set.add(paint);
  return () => {
    set!.delete(paint);
    if (set!.size === 0) painters.delete(nodeId);
  };
}

export function paintAllBoundaries(): void {
  for (const set of painters.values()) {
    for (const paint of set) paint();
  }
}

/** Paint after React Flow / child effects may have touched the DOM. */
export function paintAllBoundariesAfterCommit(): void {
  paintAllBoundaries();
  queueMicrotask(() => paintAllBoundaries());
  requestAnimationFrame(() => paintAllBoundaries());
}

export function paintBoundaryNode(nodeId: string): void {
  const set = painters.get(nodeId);
  if (!set) return;
  for (const paint of set) paint();
}
