/** Centered “starting training” chip over the flow canvas. */
export function PlayStartingOverlay({
  visible,
  label,
}: {
  visible: boolean;
  label: string;
}) {
  if (!visible) return null;
  return (
    <div className="nn-play-starting" role="status" aria-live="polite">
      <span className="nn-play-starting__spinner" aria-hidden />
      <span className="nn-play-starting__text">{label}</span>
    </div>
  );
}

/** Keep the overlay up through the first paint/commit so it covers leftover jank. */
export function dismissStartingAfterPaint(setStarting: (value: boolean) => void): void {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => setStarting(false));
  });
}
