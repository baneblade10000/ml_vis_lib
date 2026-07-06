import { Link, Navigate, useParams } from "react-router-dom";
import { NeuralNetworkPlayground, useI18n } from "@ml-vis/react";
import { LocaleSwitcher } from "../LocaleSwitcher";
import { usePlaygroundT } from "../i18n";
import { getVisualizationById } from "../visualizations";

export function VizPage() {
  const { id } = useParams<{ id: string }>();
  const { locale } = useI18n();
  const t = usePlaygroundT(locale);
  const viz = id ? getVisualizationById(id) : undefined;

  if (!viz) {
    return <Navigate to="/" replace />;
  }

  const immersive = viz.id === "neural-network";
  const backLink = (
    <Link to="/" className="catalog-back catalog-back--toolbar">
      ← {t("backToCatalog")}
    </Link>
  );

  if (immersive) {
    return (
      <div className="viz-page viz-page--immersive">
        <NeuralNetworkPlayground toolbarStart={backLink} toolbarEnd={<LocaleSwitcher />} />
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
