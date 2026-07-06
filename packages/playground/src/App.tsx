import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { I18nProvider } from "@ml-vis/react";
import { CatalogPage } from "./pages/CatalogPage";
import { VizPage } from "./pages/VizPage";
import "./tf-playground.css";

function AppContent() {
  const location = useLocation();
  const isCatalog = location.pathname === "/";

  return (
    <div className="tf-app">
      <div className={`tf-page${isCatalog ? "" : " tf-page--compact"}`}>
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
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </I18nProvider>
  );
}
