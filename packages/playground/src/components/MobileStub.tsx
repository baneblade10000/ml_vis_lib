import { useI18n } from "@ml-vis/react";
import { LocaleSwitcher } from "../LocaleSwitcher";
import { usePlaygroundT } from "../i18n";

export function MobileStub() {
  const { locale } = useI18n();
  const t = usePlaygroundT(locale);

  return (
    <div className="mobile-stub" role="status" aria-live="polite">
      <header className="mobile-stub__header">
        <p className="mobile-stub__service">{t("mobileStubService")}</p>
        <LocaleSwitcher />
      </header>

      <main className="mobile-stub__main">
        <div className="mobile-stub__emblem" aria-hidden="true">
          <svg viewBox="0 0 64 64" width="56" height="56" fill="none">
            <rect x="8" y="14" width="48" height="36" rx="3" stroke="currentColor" strokeWidth="2" />
            <path d="M8 22h48" stroke="currentColor" strokeWidth="2" />
            <circle cx="14" cy="18" r="1.5" fill="currentColor" />
            <circle cx="20" cy="18" r="1.5" fill="currentColor" />
            <circle cx="26" cy="18" r="1.5" fill="currentColor" />
            <rect x="18" y="28" width="28" height="2.5" rx="1" fill="currentColor" opacity="0.35" />
            <rect x="18" y="34" width="20" height="2.5" rx="1" fill="currentColor" opacity="0.35" />
            <rect x="18" y="40" width="24" height="2.5" rx="1" fill="currentColor" opacity="0.35" />
          </svg>
        </div>

        <h1 className="mobile-stub__title">{t("mobileStubTitle")}</h1>
        <p className="mobile-stub__body">{t("mobileStubBody")}</p>

        <div className="mobile-stub__notice">
          <p className="mobile-stub__notice-label">{t("mobileStubNoticeLabel")}</p>
          <p className="mobile-stub__notice-text">{t("mobileStubNotice")}</p>
        </div>
      </main>

      <footer className="mobile-stub__footer">
        <p>{t("mobileStubFooter")}</p>
      </footer>
    </div>
  );
}
