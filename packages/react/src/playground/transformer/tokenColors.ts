/**
 * Stable per-token color coding. The same hue follows a token through the
 * strips, the architecture diagram, and the heatmap axis labels, so attention
 * rows/columns are traceable back to their tokens at a glance.
 *
 * The vocabulary is fixed at 20 words (10 Russian + 10 English) — one distinct
 * hue per word, mirrored between the languages of the same index.
 */

const TOKEN_PALETTE = [
  "#2563eb", "#059669", "#d97706", "#7c3aed", "#e11d48",
  "#0d9488", "#db2777", "#65a30d", "#ea580c", "#4f46e5",
  // English words repeat the Russian hue with a deeper shade (index+10).
  "#1d4ed8", "#047857", "#b45309", "#6d28d9", "#be123c",
  "#0f766e", "#be185d", "#4d7c0f", "#c2410c", "#4338ca",
];

const SPECIAL = "#94a3b8";

/** Hex color for a token id; specials (`<s>`/`</s>`) get a neutral gray. */
export function tokenColor(token: number, alphabetSize: number): string {
  if (token >= alphabetSize) return SPECIAL;
  return TOKEN_PALETTE[token % TOKEN_PALETTE.length];
}

/** Soft tinted chip: background/border/text derived from the token hue. */
export function tokenChipStyle(token: number, alphabetSize: number): React.CSSProperties {
  const color = tokenColor(token, alphabetSize);
  return {
    background: `color-mix(in srgb, ${color} 14%, #ffffff)`,
    borderColor: `color-mix(in srgb, ${color} 45%, #ffffff)`,
    color: `color-mix(in srgb, ${color} 72%, #0f172a)`,
  };
}
