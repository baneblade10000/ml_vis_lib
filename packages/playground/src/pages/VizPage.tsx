import type { ComponentType, ReactNode } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import {
  ComputationalGraphPlayground,
  ConvolutionalNetworkPlayground,
  NeuralNetworkPlayground,
  useI18n,
} from "@ml-vis/react";
import { createBurnCnnTrainWorker } from "../burn/createBurnCnnTrainWorker";
import { LocaleSwitcher } from "../LocaleSwitcher";
import { usePlaygroundT } from "../i18n";
import { getVisualizationById } from "../visualizations";

const IMMERSIVE_COMPONENTS = {
  "computational-graph": ComputationalGraphPlayground,
  "neural-network": NeuralNetworkPlayground,
  "convolutional-network": ConvolutionalNetworkPlayground,
} as const;

export function VizPage() {
  const { id } = useParams<{ id: string }>();
  const { locale } = useI18n();
  const t = usePlaygroundT(locale);
  const viz = id ? getVisualizationById(id) : undefined;

  if (!viz) {
    return <Navigate to="/" replace />;
  }

  const ImmersiveComponent = IMMERSIVE_COMPONENTS[viz.id as keyof typeof IMMERSIVE_COMPONENTS];
  const immersive = ImmersiveComponent !== undefined;
  const backLink = (
    <Link to="/" className="catalog-back catalog-back--toolbar">
      ← {t("backToCatalog")}
    </Link>
  );

  if (immersive && ImmersiveComponent) {
    const isCnn = viz.id === "convolutional-network";
    // CNN needs createWorker and is rendered explicitly above; the else branch only
    // ever renders the other immersive components (toolbar props only), so narrow
    // the union away from the CNN component for this render.
    const ImmersiveToolbarComponent = ImmersiveComponent as ComponentType<{
      toolbarStart?: ReactNode;
      toolbarEnd?: ReactNode;
    }>;
    return (
      <div className="viz-page viz-page--immersive">
        {isCnn ? (
          <ConvolutionalNetworkPlayground
            toolbarStart={backLink}
            toolbarEnd={<LocaleSwitcher />}
            createWorker={createBurnCnnTrainWorker}
          />
        ) : (
          <ImmersiveToolbarComponent toolbarStart={backLink} toolbarEnd={<LocaleSwitcher />} />
        )}
      </div>
    );
  }

  const Component = viz.component;

  return (
    <div className="viz-page">
      <div className="viz-page-top">
        {backLink}
        <LocaleSwitcher />
      </div>
      <Component />
    </div>
  );
}
