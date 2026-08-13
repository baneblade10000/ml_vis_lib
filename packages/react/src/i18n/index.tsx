import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { getLocale, setLocale as setCoreLocale, subscribeLocale, type Locale } from "@ml-vis/core/i18n";

export type ReactMessages = {
  expandSection: string;
  collapseSection: string;
  sectionsNav: string;
};

export const reactMessages: Record<Locale, ReactMessages> = {
  en: {
    expandSection: "Expand section",
    collapseSection: "Collapse section",
    sectionsNav: "Sections",
  },
  ru: {
    expandSection: "Развернуть секцию",
    collapseSection: "Свернуть секцию",
    sectionsNav: "Секции",
  },
};

export type I18nContextValue = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: keyof ReactMessages) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

export type I18nProviderProps = {
  children: ReactNode;
  defaultLocale?: Locale;
};

export function I18nProvider({ children, defaultLocale = "ru" }: I18nProviderProps) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    setCoreLocale(defaultLocale);
    return defaultLocale;
  });

  useEffect(() => {
    return subscribeLocale(() => setLocaleState(getLocale()));
  }, []);

  const setLocale = (next: Locale) => {
    setCoreLocale(next);
  };

  const value = useMemo<I18nContextValue>(
    () => ({
      locale,
      setLocale,
      t: (key) => reactMessages[locale][key],
    }),
    [locale],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error("useI18n must be used within I18nProvider");
  }
  return ctx;
}

export function useLocale(): Locale {
  const [locale, setLocaleState] = useState(getLocale);

  useEffect(() => subscribeLocale(() => setLocaleState(getLocale())), []);

  return locale;
}

export { type Locale } from "@ml-vis/core/i18n";
export { getLocale, localeLabels, locales, setLocale, subscribeLocale } from "@ml-vis/core/i18n";
