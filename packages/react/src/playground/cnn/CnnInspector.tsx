import { useLayoutEffect, useRef } from "react";
import {
  renderValueMatrix,
  weightColor,
  weightValueNormalized,
} from "@ml-vis/core";
import { useCnnMessages } from "./messages";

export interface CnnInspectorProps {
  selectedLayerId: string | null;
  /** Kernel display snapshots: layerId → array of per-filter kernel maps/vectors. */
  kernels: Record<string, number[][] | number[][][]>;
  /** Per-filter biases for conv layers: layerId → biases[filter]. */
  biases?: Record<string, number[]>;
  info: {
    kind: string;
    label: string;
    inputShape: string;
    outputShape: string;
    params: number;
  } | null;
}

/** Map kernel cells onto the same tanh-normalized scale as NN edge weights. */
function normalizeWeightMap(map: number[][]): number[][] {
  return map.map((row) => row.map((w) => weightValueNormalized(w)));
}

function KernelThumb({
  map,
  size,
  bias,
}: {
  map: number[][];
  size: number;
  bias?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatRef = useRef<HTMLCanvasElement>(null);
  const cols = map[0]?.length ?? 1;
  const rows = map.length;

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const heat = heatRef.current;
    if (!canvas || !heat || !map.length) return;
    renderValueMatrix(heat, normalizeWeightMap(map), {
      layout: "row-major",
      palette: "diverging",
    });
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(heat, 0, 0, size, size);
  }, [map, size]);

  const title =
    typeof bias === "number" ? `bias ${bias.toFixed(3)}` : undefined;

  return (
    <div className="cnn-kernel-with-bias" title={title}>
      {typeof bias === "number" && (
        <span
          className="cnn-filter-bias"
          data-sign={bias >= 0 ? "pos" : "neg"}
          aria-hidden
          style={{ background: weightColor(weightValueNormalized(bias)) }}
        />
      )}
      <div className="cnn-kernel-thumb" style={{ width: size, height: size }}>
        <canvas ref={heatRef} width={cols} height={rows} hidden aria-hidden />
        <canvas ref={canvasRef} width={size} height={size} className="cnn-feature-canvas" />
      </div>
    </div>
  );
}

function KernelGrid2D({ maps, biases }: { maps: number[][][]; biases?: number[] }) {
  return (
    <div className="cnn-kernel-grid">
      {maps.map((m, i) => (
        <KernelThumb key={i} map={m} size={36} bias={biases?.[i]} />
      ))}
    </div>
  );
}

function KernelGrid1D({ vectors, biases }: { vectors: number[][]; biases?: number[] }) {
  return (
    <div className="cnn-kernel-grid cnn-kernel-grid--1d">
      {vectors.map((v, i) => (
        <KernelThumb key={i} map={[v]} size={72} bias={biases?.[i]} />
      ))}
    </div>
  );
}

/** Inspector for the selected layer: shapes, param count, and conv kernels. */
export function CnnInspector({ selectedLayerId, kernels, biases, info }: CnnInspectorProps) {
  const t = useCnnMessages();

  if (!selectedLayerId || !info) {
    return (
      <div className="cnn-inspector">
        <h4 className="nn-flow-dock-title">{t.inspectorKind}</h4>
        <p className="nn-arch-hint">{t.inspectorEmpty}</p>
      </div>
    );
  }

  const kernelData = kernels[selectedLayerId];
  const layerBiases = biases?.[selectedLayerId];
  const is2d = kernelData != null && Array.isArray((kernelData as number[][][])[0]?.[0]);
  const is1d = kernelData != null && !is2d;

  return (
    <div className="cnn-inspector">
      <h4 className="nn-flow-dock-title">{info.label}</h4>
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
          <span className="nn-flow-dock-title">{t.inspectorKernels}</span>
          <KernelGrid2D maps={kernelData as number[][][]} biases={layerBiases} />
        </>
      )}
      {is1d && (
        <>
          <span className="nn-flow-dock-title">{t.inspectorKernels}</span>
          <KernelGrid1D vectors={kernelData as number[][]} biases={layerBiases} />
        </>
      )}
      {layerBiases && layerBiases.length > 0 && (
        <>
          <span className="nn-flow-dock-title">{t.inspectorBiases}</span>
          <div className="cnn-bias-list">
            {layerBiases.map((b, i) => (
              <div key={i} className="cnn-bias-list__row">
                <span
                  className="cnn-filter-bias cnn-filter-bias--inline"
                  data-sign={b >= 0 ? "pos" : "neg"}
                  aria-hidden
                  style={{ background: weightColor(weightValueNormalized(b)) }}
                />
                <span className="cnn-bias-list__label">
                  {t.inspectorBias} {i + 1}
                </span>
                <span className="cnn-bias-list__val">{b.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
