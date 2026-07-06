import { Link } from "react-router-dom";
import { useI18n } from "@ml-vis/react";
import { LocaleSwitcher } from "../LocaleSwitcher";
import { usePlaygroundT } from "../i18n";
import { visualizations } from "../visualizations";

export function CatalogPage() {
  const { locale } = useI18n();
  const t = usePlaygroundT(locale);

  return (
    <div className="catalog-page">
      <div className="catalog-intro">
        <div className="catalog-intro-head">
          <h2>{t("catalogTitle")}</h2>
          <LocaleSwitcher />
        </div>
        <p>{t("catalogDescription")}</p>
      </div>
      <ul className="catalog-grid">
        {visualizations.map((viz) => (
          <li key={viz.id}>
            <Link to={viz.path} className="catalog-card">
              <div className="catalog-card-body">
                <h3>{t(viz.titleKey)}</h3>
                <p>{t(viz.descriptionKey)}</p>
              </div>
              <span className="catalog-card-action">{t("openVisualization")}</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
