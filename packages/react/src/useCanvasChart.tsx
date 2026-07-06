import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

export function useCanvasChart<T extends { resize?: () => void; destroy: () => void }>(
  create: (canvas: HTMLCanvasElement) => T,
  destroy: (chart: T) => void,
  createDeps: unknown[],
  sync?: (chart: T) => void,
  syncDeps: unknown[] = [],
) {
  const chartRef = useRef<T | null>(null);
  const syncRef = useRef(sync);
  syncRef.current = sync;
  const [canvasNode, setCanvasNode] = useState<HTMLCanvasElement | null>(null);

  const canvasRef = useCallback((node: HTMLCanvasElement | null) => {
    setCanvasNode(node);
  }, []);

  useEffect(() => {
    if (!canvasNode) return;

    const chart = create(canvasNode);
    chartRef.current = chart;
    syncRef.current?.(chart);

    const observer = new ResizeObserver(() => {
      chartRef.current?.resize?.();
    });
    observer.observe(canvasNode);

    return () => {
      observer.disconnect();
      destroy(chart);
      chartRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canvasNode, ...createDeps]);

  useEffect(() => {
    if (chartRef.current) syncRef.current?.(chartRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, syncDeps);

  return { canvasRef, chartRef };
}

export function ChartBox({
  height = 360,
  aspectRatio,
  onWidth,
  children,
}: {
  height?: number;
  aspectRatio?: number;
  onWidth?: (width: number) => void;
  children: (width: number) => ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState<number | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const next = Math.floor(el.clientWidth);
      if (next <= 0) return;
      setWidth(next);
      onWidth?.(next);
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, [onWidth]);

  return (
    <div
      ref={ref}
      style={{
        width: "100%",
        ...(aspectRatio != null ? { aspectRatio: String(aspectRatio) } : { minHeight: height }),
      }}
    >
      {width !== null ? children(width) : null}
    </div>
  );
}
