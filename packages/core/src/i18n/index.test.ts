import { describe, expect, it } from "vitest";
import { getLocale, setLocale, t } from "./index";

describe("i18n", () => {
  it("returns Russian messages by default", () => {
    expect(getLocale()).toBe("ru");
    expect(t("noData")).toBe("Нет данных");
  });

  it("switches to English messages", () => {
    setLocale("en");
    expect(getLocale()).toBe("en");
    expect(t("noData")).toBe("No data");
    setLocale("ru");
  });
});
