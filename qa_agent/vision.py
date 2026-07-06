from __future__ import annotations

import base64
import logging
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

from PIL import Image, ImageStat

logger = logging.getLogger("qa_agent.vision")

_DOM_NODE_HINTS = (
    "rf__node",
    "react-flow",
    "tf-flow-node",
    "canvas",
    "heatmap",
    "decision boundary",
)
_DOM_OUTPUT_HINTS = ("loss-test", "loss-train", "tf-output", "tf-stat-value")


@dataclass
class VisualDigest:
    """Machine-readable summary of what the screenshot actually shows."""

    summary: str
    mostly_blank: bool
    blank_score: float
    dom_mismatches: list[str]
    vision_description: str | None = None
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _downsample(img: Image.Image, size: int = 96) -> Image.Image:
    return img.resize((size, size), Image.Resampling.BILINEAR)


def analyze_blank_heuristic(png_bytes: bytes) -> tuple[bool, float, str]:
    """Estimate whether the viewport center is an empty / uniform canvas."""
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    left = int(w * 0.2)
    top = int(h * 0.12)
    right = int(w * 0.82)
    bottom = int(h * 0.88)
    crop = img.crop((left, top, right, bottom))
    sample = _downsample(crop, 96)
    stat = ImageStat.Stat(sample)
    mean_std = sum(stat.stddev) / 3.0
    colors = sample.getcolors(maxcolors=256 * 256)
    unique = len(colors) if colors else 256 * 256

    blank_score = 0.0
    if mean_std < 6:
        blank_score += 0.55
    elif mean_std < 12:
        blank_score += 0.35
    if unique <= 8:
        blank_score += 0.35
    elif unique <= 16:
        blank_score += 0.2
    blank_score = min(1.0, blank_score)
    mostly_blank = blank_score >= 0.65
    note = (
        f"center_std={mean_std:.1f}, unique_colors={unique}, blank_score={blank_score:.2f}"
    )
    return mostly_blank, blank_score, note


def detect_dom_mismatches(dom_digest: str, mostly_blank: bool) -> list[str]:
    """Flag when DOM claims rich UI but the screenshot looks empty."""
    if not mostly_blank:
        return []

    lowered = dom_digest.lower()
    mismatches: list[str] = []
    if any(hint in lowered for hint in _DOM_NODE_HINTS):
        mismatches.append(
            "DOM lists graph/canvas nodes but the screenshot center looks empty or uniform"
        )
    if any(hint in lowered for hint in _DOM_OUTPUT_HINTS):
        mismatches.append(
            "DOM lists output metrics/charts but the screenshot center looks empty"
        )
    rf_nodes = len(re.findall(r"rf__node", dom_digest))
    if rf_nodes >= 2:
        mismatches.append(f"DOM has {rf_nodes} React Flow nodes but they are not visible")
    return mismatches


def _build_summary(
    *,
    mostly_blank: bool,
    blank_score: float,
    dom_mismatches: list[str],
    vision_description: str | None,
    blank_note: str,
) -> str:
    parts: list[str] = []
    if mostly_blank:
        parts.append("Screenshot center looks mostly empty/uniform (possible broken render).")
    else:
        parts.append("Screenshot center shows visible UI content.")
    if dom_mismatches:
        parts.append(" ".join(dom_mismatches))
    if vision_description:
        parts.append(vision_description.strip())
    else:
        parts.append(f"Heuristic: {blank_note}.")
    return " ".join(parts)


def analyze_screenshot_heuristic(
    png_bytes: bytes,
    *,
    dom_digest: str,
) -> VisualDigest:
    """Local blank-canvas / DOM mismatch checks (no LLM)."""
    mostly_blank, blank_score, blank_note = analyze_blank_heuristic(png_bytes)
    dom_mismatches = detect_dom_mismatches(dom_digest, mostly_blank)
    summary = _build_summary(
        mostly_blank=mostly_blank,
        blank_score=blank_score,
        dom_mismatches=dom_mismatches,
        vision_description=None,
        blank_note=blank_note,
    )
    return VisualDigest(
        summary=summary,
        mostly_blank=mostly_blank,
        blank_score=blank_score,
        dom_mismatches=dom_mismatches,
        source="heuristic",
    )


def analyze_screenshot_bytes(
    png_bytes: bytes,
    *,
    dom_digest: str,
    screenshot_b64: str = "",
) -> VisualDigest:
    _ = screenshot_b64
    return analyze_screenshot_heuristic(png_bytes, dom_digest=dom_digest)
