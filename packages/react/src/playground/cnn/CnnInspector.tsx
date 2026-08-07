import { useLayoutEffect, useRef } from "react";
import { renderValueMatrix } from "@ml-vis/core";
import { useCnnMessages } from "./messages";

export interface CnnInspectorProps {
  selectedLayerId: string | null;
  /** Kernel display snapshots: layerId → array of per-filter kernel maps/vectors. */
  kernels: Record<string, number[][] | number[][][]>;
  info: {
    kind: string;
    label: string;
    inputShape: string;
    outputShape: string;
    params: number;
  } | null;
}

function KernelThumb({ map, size }: { map: number[][]; size: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatRef = useRef<HTMLCanvasElement>(null);
  const cols = map[0]?.length ?? 1;
  const rows = map.length;

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const heat = heatRef.current;
    if (!canvas || !heat || !map.length) return;
    renderValueMatrix(heat, map, { layout: "row-major", palette: "gray" });
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(heat, 0, 0, size, size);
  }, [map, size]);

  return (
    <div className="cnn-kernel-thumb" style={{ width: size, height: size }}>
      <canvas ref={heatRef} width={cols} height={rows} hidden aria-hidden />
      <canvas ref={canvasRef} width={size} height={size} className="cnn-feature-canvas" />
    </div>
  );
}

function KernelGrid2D({ maps }: { maps: number[][][] }) {
  return (
    <div className="cnn-kernel-grid">
      {maps.map((m, i) => (
        <KernelThumb key={i} map={m} size={36} />
      ))}
    </div>
  );
}

function KernelGrid1D({ vectors }: { vectors: number[][] }) {
  return (
    <div className="cnn-kernel-grid cnn-kernel-grid--1d">
      {vectors.map((v, i) => (
        <KernelThumb key={i} map={[v]} size={72} />
      ))}
    </div>
  );
}

/** Inspector for the selected layer: shapes, param count, and conv kernels. */
export function CnnInspector({ selectedLayerId, kernels, info }: CnnInspectorProps) {
  const t = useCnnMessages();

  if (!selectedLayerId || !info) {
    return (
      <div className="cnn-inspector">
        <h4 className="tf-flow-dock-title">{t.inspectorKind}</h4>
        <p className="tf-arch-hint">{t.inspectorEmpty}</p>
      </div>
    );
  }

  const kernelData = kernels[selectedLayerId];
  const is2d = kernelData != null && Array.isArray((kernelData as number[][][])[0]?.[0]);
  const is1d = kernelData != null && !is2d;

  return (
    <div className="cnn-inspector">
      <h4 className="tf-flow-dock-title">{info.label}</h4>
      <div className="cnn-inspector-rows">
        <div className="cnn-inspector-row">
          <span className="cnn-inspector-key">{t.inspectorKind}</span>
          <span className="cnn-inspector-val">{info.kind}</span>
        </div>
        <div className="cnn-inspector-row">
          <span className="cnn-inspector-key">{t.inspectorInputShape}</span>
          <span className="cnn-inspector-val">{info.inputShape}</span>
        </div>
        <div className="cnn-inspector-row">
          <span className="cnn-inspector-key">{t.inspectorOutputShape}</span>
          <span className="cnn-inspector-val">{info.outputShape}</span>
        </div>
        <div className="cnn-inspector-row">
          <span className="cnn-inspector-key">{t.inspectorParams}</span>
          <span className="cnn-inspector-val">{info.params}</span>
        </div>
      </div>
      {is2d && (
        <>
          <span className="tf-flow-dock-title">{t.inspectorKernels}</span>
          <KernelGrid2D maps={kernelData as number[][][]} />
        </>
      )}
      {is1d && (
        <>
          <span className="tf-flow-dock-title">{t.inspectorKernels}</span>
          <KernelGrid1D vectors={kernelData as number[][]} />
        </>
      )}
    </div>
  );
}
