import { HashRouter, Route, Routes, useLocation } from "react-router-dom";
import { I18nProvider } from "@ml-vis/react";
import { MobileStub } from "./components/MobileStub";
import { useIsMobile } from "./hooks/useIsMobile";
import { CatalogPage } from "./pages/CatalogPage";
import { VizPage } from "./pages/VizPage";
import "./network-playground.css";
import "./transformer-playground.css";

function AppContent() {
  const location = useLocation();
  const isCatalog = location.pathname === "/";
  const isMobile = useIsMobile();

  if (isMobile) {
    return <MobileStub />;
  }

  return (
    <div className="nn-app">
      <div className={`nn-page${isCatalog ? "" : " nn-page--compact"}`}>
        <Routes>
          <Route path="/" element={<CatalogPage />} />
          <Route path="/viz/:id" element={<VizPage />} />
        </Routes>
      </div>
    </div>
  );
}

export function App() {
  return (
    <I18nProvider>
      {/* HashRouter: works on static hosts without SPA rewrite (jsDelivr / Pages). */}
      <HashRouter>
        <AppContent />
      </HashRouter>
    </I18nProvider>
  );
}
