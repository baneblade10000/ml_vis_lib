import { tokenChipStyle } from "./tokenColors";

export interface TokenStripProps {
  tokens: number[];
  /** Display label per token id (from the snapshot). */
  labels: string[];
  tone?: "input" | "target" | "prediction";
  /** For `prediction`: expected tokens to mark correct/incorrect positions. */
  expected?: number[];
}

export function TokenStrip({ tokens, labels, tone = "input", expected }: TokenStripProps) {
  return (
    <div className={`tf-tokens tf-tokens--${tone}`}>
      {tokens.map((tok, i) => {
        const special = tok >= labels.length - 2;
        const correct = expected && i < expected.length ? expected[i] === tok : null;
        return (
          <span
            key={i}
            className={`tf-token${special ? " tf-token--special" : ""}${
              correct === true ? " tf-token--ok" : correct === false ? " tf-token--bad" : ""
            }`}
            style={special ? undefined : tokenChipStyle(tok, labels.length - 2)}
          >
            {labels[tok] ?? "?"}
            {correct === true && <i className="tf-token__mark" aria-hidden>✓</i>}
            {correct === false && <i className="tf-token__mark" aria-hidden>✗</i>}
          </span>
        );
      })}
      {tokens.length === 0 && <span className="tf-token tf-token--special">∅</span>}
    </div>
  );
}
