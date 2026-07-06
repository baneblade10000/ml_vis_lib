export type Locale = "en" | "ru";

export type CoreMessages = {
  noData: string;
};

export const coreMessages: Record<Locale, CoreMessages> = {
  en: {
    noData: "No data",
  },
  ru: {
    noData: "Нет данных",
  },
};

export const locales: Locale[] = ["en", "ru"];

export const localeLabels: Record<Locale, string> = {
  en: "English",
  ru: "Русский",
};

let currentLocale: Locale = "ru";
const listeners = new Set<() => void>();

export function getLocale(): Locale {
  return currentLocale;
}

export function setLocale(locale: Locale): void {
  if (currentLocale === locale) return;
  currentLocale = locale;
  for (const listener of listeners) {
    listener();
  }
}

export function subscribeLocale(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function t(key: keyof CoreMessages): string {
  return coreMessages[currentLocale][key];
}
