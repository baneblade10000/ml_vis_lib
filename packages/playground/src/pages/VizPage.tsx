import { useEffect, useRef, useState, type ComponentType, type ReactNode } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import {
  ComputationalGraphPlayground,
  ConvolutionalNetworkPlayground,
  NeuralNetworkPlayground,
  TransformerPlayground,
  useI18n,
} from "@ml-vis/react";
import { createCnnTrainWorker } from "../wasm/cnn/createCnnTrainWorker";
import { createTransformerTrainWorker } from "../wasm/transformer/createTransformerTrainWorker";
import { LocaleSwitcher } from "../LocaleSwitcher";
import { usePlaygroundT } from "../i18n";
import {
  getVisualizationById,
  type EmbeddedPage,
  type EmbeddedPageHeaderMessage,
} from "../visualizations";

const IMMERSIVE_COMPONENTS = {
  "computational-graph": ComputationalGraphPlayground,
  "neural-network": NeuralNetworkPlayground,
  "convolutional-network": ConvolutionalNetworkPlayground,
  transformer: TransformerPlayground,
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
    // CNN/transformer need createWorker and are rendered explicitly below; the
    // else branch only renders the other immersive components (toolbar props
    // only), so narrow the union for that render.
    const ImmersiveToolbarComponent = ImmersiveComponent as ComponentType<{
      toolbarStart?: ReactNode;
      toolbarEnd?: ReactNode;
    }>;
    return (
      <div className="viz-page viz-page--immersive">
        {viz.id === "convolutional-network" ? (
          <ConvolutionalNetworkPlayground
            toolbarStart={backLink}
            toolbarEnd={<LocaleSwitcher />}
            createWorker={createCnnTrainWorker}
          />
        ) : viz.id === "transformer" ? (
          <TransformerPlayground
            toolbarStart={backLink}
            toolbarEnd={<LocaleSwitcher />}
            createWorker={createTransformerTrainWorker}
          />
        ) : (
          <ImmersiveToolbarComponent toolbarStart={backLink} toolbarEnd={<LocaleSwitcher />} />
        )}
      </div>
    );
  }

  if (viz.embeddedPage) {
    return <EmbeddedVizPage page={viz.embeddedPage} backLink={backLink} />;
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

/**
 * A page from public/ has no React toolbar: its tabs sit in the catalog header
 * and are forwarded to the page, which drops its own header when embedded. The
 * page reports the tab it switched to on its own, and the title and badge its
 * header shows; its actions are forwarded like the tabs.
 */
function EmbeddedVizPage({ page, backLink }: { page: EmbeddedPage; backLink: ReactNode }) {
  const { locale } = useI18n();
  const t = usePlaygroundT(locale);
  const [tab, setTab] = useState(page.defaultTab ?? page.tabs?.[0]?.id);
  const [header, setHeader] = useState<Pick<EmbeddedPageHeaderMessage, "title" | "badge">>({});
  const frameRef = useRef<HTMLIFrameElement>(null);

  const post = (message: object) => frameRef.current?.contentWindow?.postMessage(message, "*");

  const showTab = (id: string | undefined) => {
    if (!id) return;
    setTab(id);
    post({ type: "ml-vis:tab", id });
  };

  useEffect(() => {
    const onMessage = (event: MessageEvent<EmbeddedPageHeaderMessage>) => {
      const frame = frameRef.current;
      if (!frame || event.source !== frame.contentWindow || event.origin !== window.location.origin) {
        return;
      }
      if (event.data?.type !== "ml-vis:header") return;
      setHeader({ title: event.data.title, badge: event.data.badge });
      if (event.data.tab) setTab(event.data.tab);
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  return (
    <div className="viz-page">
      <div className="viz-page-top">
        {backLink}
        {header.title && <h2 className="viz-page-title">{header.title}</h2>}
        {page.tabs && (
          <div className="viz-tabs" role="tablist">
            {page.tabs.map((item) => (
              <button
                key={item.id}
                type="button"
                role="tab"
                aria-selected={item.id === tab}
                className={`viz-tab${item.id === tab ? " selected" : ""}`}
                onClick={() => showTab(item.id)}
              >
                {t(item.labelKey)}
              </button>
            ))}
          </div>
        )}
        {header.badge && <span className="viz-page-badge">{header.badge}</span>}
        {page.actions && (
          <div className="viz-page-actions">
            {page.actions.map((item) => (
              <button
                key={item.id}
                type="button"
                className="nn-btn nn-btn--ghost viz-page-action"
                onClick={() => post({ type: "ml-vis:action", id: item.id })}
              >
                {t(item.labelKey)}
              </button>
            ))}
          </div>
        )}
        <LocaleSwitcher />
      </div>
      <iframe
        ref={frameRef}
        className="public-page-frame"
        src={`${import.meta.env.BASE_URL}${page.src}`}
        title={page.title}
        // The page only hears its tab once loaded; a resend covers reloads too.
        onLoad={() => showTab(tab)}
      />
    </div>
  );
}
