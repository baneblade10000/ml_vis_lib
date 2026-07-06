from __future__ import annotations

from io import BytesIO

from PIL import Image

from qa_agent.vision import analyze_blank_heuristic, detect_dom_mismatches


def _solid_png(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (800, 600), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_blank_heuristic_detects_uniform_canvas() -> None:
    mostly_blank, score, _ = analyze_blank_heuristic(_solid_png((240, 240, 240)))
    assert mostly_blank is True
    assert score >= 0.65


def test_analyze_blank_heuristic_detects_varied_content() -> None:
    img = Image.new("RGB", (800, 600), (255, 255, 255))
    for x in range(0, 800, 40):
        for y in range(0, 600, 40):
            img.putpixel((x, y), (x % 255, y % 255, (x + y) % 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    mostly_blank, score, _ = analyze_blank_heuristic(buf.getvalue())
    assert mostly_blank is False
    assert score < 0.65


def test_detect_dom_mismatches_when_blank() -> None:
    dom = "rf__node-input rf__node-hidden rf__node-output loss-test"
    mismatches = detect_dom_mismatches(dom, mostly_blank=True)
    assert mismatches
    assert any("React Flow" in item for item in mismatches)
