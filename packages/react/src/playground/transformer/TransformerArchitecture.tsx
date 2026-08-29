import type { TransformerSnapshot } from "@ml-vis/core/transformer";
import type { AttentionKind } from "./TransformerPlayground";
import { useTransformerMessages } from "./messages";
import { TokenStrip } from "./TokenStrip";

export interface TransformerArchitectureProps {
  snapshot: TransformerSnapshot;
  focus: { kind: AttentionKind; layer: number } | null;
  onFocusChange: (next: { kind: AttentionKind; layer: number } | null) => void;
}

/**
 * Encoder/decoder stack diagram. Attention blocks are clickable and select the
 * matching heatmap layer; cross-attention blocks carry the memory accent.
 */
export function TransformerArchitecture({ snapshot, focus, onFocusChange }: TransformerArchitectureProps) {
  const t = useTransformerMessages();
  const heads = snapshot.attention.encSelf[0]?.length ?? 0;
  const block = (kind: AttentionKind, layer: number, label: string, extra?: string) => {
    const selected = focus?.kind === kind && focus.layer === layer;
    return (
      <button
        type="button"
        className={`tf-block tf-block--attention${kind === "cross" ? " tf-block--cross" : ""}${selected ? " selected" : ""}`}
        onClick={() => onFocusChange(selected ? null : { kind, layer })}
      >
        <span className="tf-block__label">{label}</span>
        <span className="tf-block__meta">
          {heads > 0 && `×${heads}`}
          {extra ? ` · ${extra}` : ""}
        </span>
      </button>
    );
  };

  return (
    <div className="tf-arch">
      <div className="tf-arch__col">
        <div className="tf-arch__head">{t.encoder}</div>
        <div className="tf-arch__tokens">
          <TokenStrip tokens={snapshot.encInTokens} labels={snapshot.labels} />
        </div>
        {[...snapshot.attention.encSelf].reverse().map((_, idx) => {
          const layer = snapshot.attention.encSelf.length - 1 - idx;
          return (
            <div key={layer} className="tf-arch__layer">
              {block("encSelf", layer, t.selfAttention, `L${layer + 1}`)}
              <div className="tf-block tf-block--plain">
                <span className="tf-block__label">{t.feedForward}</span>
                <span className="tf-block__meta">L{layer + 1}</span>
              </div>
            </div>
          );
        })}
        <div className="tf-arch__memory">
          <span className="tf-arch__memory-label">{t.memory}</span>
          <span className="tf-arch__memory-meta">{snapshot.encInTokens.length} × d</span>
        </div>
      </div>

      <div className="tf-arch__bridge" aria-hidden />

      <div className="tf-arch__col">
        <div className="tf-arch__head">{t.decoder}</div>
        <div className="tf-arch__tokens">
          <TokenStrip tokens={snapshot.decInTokens} labels={snapshot.labels} />
        </div>
        {[...snapshot.attention.decSelf].reverse().map((_, idx) => {
          const layer = snapshot.attention.decSelf.length - 1 - idx;
          return (
            <div key={layer} className="tf-arch__layer">
              {block("decSelf", layer, t.maskedSelfAttention, `L${layer + 1}`)}
              {block("cross", layer, t.crossAttention, `L${layer + 1}`)}
              <div className="tf-block tf-block--plain">
                <span className="tf-block__label">{t.feedForward}</span>
                <span className="tf-block__meta">L{layer + 1}</span>
              </div>
            </div>
          );
        })}
        <div className="tf-arch__out">
          <span className="tf-arch__memory-label">{t.logits}</span>
          <TokenStrip
            tokens={snapshot.predictedTokens}
            labels={snapshot.labels}
            tone="prediction"
            expected={snapshot.targetTokens}
          />
        </div>
        <span className="tf-arch__axis-hint">{t.decoder.toLowerCase()} → {t.logits.toLowerCase()}</span>
      </div>
    </div>
  );
}
