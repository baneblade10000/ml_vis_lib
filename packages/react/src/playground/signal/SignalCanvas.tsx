import { useRef, useState } from "react";
import { SignalPlot, type SignalPlotPayload } from "@ml-vis/core/charts";
import { ChartBox, useCanvasChart } from "../../useCanvasChart";

interface SignalCanvasProps {
  payload: SignalPlotPayload;
  height?: number;
}

/** Binds a {@link SignalPlot} to a canvas via {@link useCanvasChart}. */
export function SignalCanvas({ payload, height = 420 }: SignalCanvasProps) {
  const plotRef = useRef<SignalPlot | null>(null);
  const [width, setWidth] = useState(0);

  const { canvasRef } = useCanvasChart(
    (canvas) => {
      const plot = new SignalPlot(canvas);
      plotRef.current = plot;
      return plot;
    },
    (plot) => plot.destroy(),
    [height],
    (plot) => {
      if (width > 0) plot.setSize(width, height);
      plot.setData(payload);
    },
    [payload, width, height],
  );

  return (
    <ChartBox height={height} onWidth={setWidth}>
      {(measuredWidth) => (
        <canvas
          ref={canvasRef}
          style={{ width: measuredWidth, height, display: "block" }}
        />
      )}
    </ChartBox>
  );
}
