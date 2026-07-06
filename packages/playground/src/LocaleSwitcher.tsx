import { localeLabels, locales, useI18n } from "@ml-vis/react";
import { usePlaygroundT } from "./i18n";

export function LocaleSwitcher({ className }: { className?: string }) {
  const { locale, setLocale } = useI18n();
  const t = usePlaygroundT(locale);

  return (
    <label className={className ? `locale-switcher ${className}` : "locale-switcher"}>
      {t("language")}:
      <select value={locale} onChange={(e) => setLocale(e.target.value as typeof locale)}>
        {locales.map((code) => (
          <option key={code} value={code}>
            {localeLabels[code]}
          </option>
        ))}
      </select>
    </label>
  );
}
