import { useState } from "react";
import { Link } from "react-router-dom";
import { useI18n } from "@ml-vis/react";
import { LocaleSwitcher } from "../LocaleSwitcher";
import { usePlaygroundT } from "../i18n";
import { visualizations } from "../visualizations";
import { CatalogTopicGraph } from "../catalog/CatalogTopicGraph";
import { searchCatalog } from "../catalog/search";

type CatalogMode = "graph" | "list";

export function CatalogPage() {
  const { locale } = useI18n();
  const t = usePlaygroundT(locale);
  const [mode, setMode] = useState<CatalogMode>("graph");
  const [query, setQuery] = useState("");

  return (
    <div className="catalog-page">
      <div className="catalog-intro">
        <div className="catalog-intro-head">
          <h2>{t("catalogTitle")}</h2>
          <input
            type="search"
            className="catalog-search"
            placeholder={t("catalogSearchPlaceholder")}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") setQuery("");
            }}
          />
          <div className="catalog-mode-switch">
            <button
              type="button"
              className={`catalog-mode-option${mode === "graph" ? " selected" : ""}`}
              aria-pressed={mode === "graph"}
              onClick={() => setMode("graph")}
            >
              {t("catalogViewGraph")}
            </button>
            <button
              type="button"
              className={`catalog-mode-option${mode === "list" ? " selected" : ""}`}
              aria-pressed={mode === "list"}
              onClick={() => setMode("list")}
            >
              {t("catalogViewList")}
            </button>
          </div>
          <LocaleSwitcher />
        </div>
      </div>
      {mode === "graph" ? <CatalogTopicGraph query={query} /> : <CatalogCardGrid query={query} />}
    </div>
  );
}

function CatalogCardGrid({ query }: { query: string }) {
  const { locale } = useI18n();
  const t = usePlaygroundT(locale);
  const search = searchCatalog(locale, query);
  const shown = visualizations.filter((viz) => !search.searching || search.lit.has(viz.id));

  return (
    <>
      <ul className="catalog-grid">
        {shown.map((viz) => (
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
      {search.searching && shown.length === 0 && (
        <p className="catalog-list-empty">{t("catalogSearchEmpty")}</p>
      )}
    </>
  );
}
