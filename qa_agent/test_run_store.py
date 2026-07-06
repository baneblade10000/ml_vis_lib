from __future__ import annotations

import json
from pathlib import Path

from qa_agent.run_compare import compare_runs
from qa_agent.run_store import list_disk_runs, resolve_run_dir, summarize_disk_run


def test_list_disk_runs_finds_existing_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("qa_agent.run_store._RUNS_ROOT", tmp_path)
    run_id = "20260630T120000Z-abc12345"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "run_start",
                "run_id": run_id,
                "ts": "2026-06-30T12:00:00+00:00",
                "data": {"goal": "test", "max_steps": 5, "exploration_profile": "learn"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runs = list_disk_runs(limit=5)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "running"


def test_compare_runs_delta(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("qa_agent.run_store._RUNS_ROOT", tmp_path)

    def _write_run(run_id: str, score: float, visited: list[str]) -> None:
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        metrics = {
            "overall_score": score,
            "key_route_coverage": 50.0,
            "visited_areas": visited,
            "missing_areas": [],
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (run_dir / "events.jsonl").write_text(
            json.dumps({"event": "run_complete", "run_id": run_id, "data": {}}) + "\n",
            encoding="utf-8",
        )

    _write_run("20260630T120000Z-aaaaaaaa", 70.0, ["Learn · modules list"])
    _write_run("20260630T130000Z-bbbbbbbb", 80.0, ["Learn · modules list", "Learn · quiz"])

    result = compare_runs("20260630T120000Z-aaaaaaaa", "20260630T130000Z-bbbbbbbb")
    assert result["score_delta"] == 10.0
    assert result["coverage_improved"] == ["Learn · quiz"]


def test_resolve_run_dir_rejects_invalid_ids(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("qa_agent.run_store._RUNS_ROOT", tmp_path)
    assert resolve_run_dir("../etc/passwd") is None
    assert resolve_run_dir("not-a-run-id") is None
