import type { Meta, StoryObj } from "@storybook/react";
import { I18nProvider } from "../i18n";
import { Section, SectionLayout, SectionNav, SectionProvider } from "./index";

const sections = [
  { id: "main", title: "Main", size: "lg" as const, order: 0 },
  { id: "info", title: "Info", size: "sm" as const, order: 1 },
];

const meta: Meta = {
  title: "Layout/Sections",
  parameters: { layout: "padded" },
};

export default meta;
type Story = StoryObj;

export const Dashboard: Story = {
  render: () => (
    <I18nProvider defaultLocale="en">
      <SectionProvider sections={sections}>
        <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: 16 }}>
          <SectionNav />
          <SectionLayout columns={2}>
            <Section id="main" title="Main panel" size="lg">
              <p style={{ margin: 0, fontSize: 14, color: "#64748b" }}>Chart slot</p>
            </Section>
            <Section id="info" title="Info" size="sm">
              <p style={{ margin: 0, fontSize: 14, color: "#64748b" }}>Side panel</p>
            </Section>
          </SectionLayout>
        </div>
      </SectionProvider>
    </I18nProvider>
  ),
};
