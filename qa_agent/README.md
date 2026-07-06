# QA Exploration Agent

Autonomous AI browser tester for the **ml-vis** playground (`@ml-vis/playground`). The agent uses Playwright to explore the Decision Boundary demo, **Cursor MCP sampling** (built-in vision LLM) for screenshot analysis and action choice, and writes a markdown bug report with screenshots and UX metrics.

## Setup

From the monorepo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r qa_agent/requirements.txt
playwright install chromium
```

Start the playground before running the agent:

```bash
pnpm install
pnpm dev
```

Playground URL: http://localhost:5173

## Usage (MCP in Cursor)

Exploration runs are **MCP-only** from Cursor. Vision and actions use **Cursor MCP sampling** — no external LLM API, no manual PNG reading in chat.

1. Reload MCP server `ml-vis-qa-agent` (see `.cursor/mcp.json`)
2. Enable MCP sampling for the server in Cursor settings
3. Ask: *"Run QA exploration using qa-cursor-driver prompt"*

**Stepwise (recommended):**

```
qa_cursor_start(profile="full", mobile=false, max_steps=45)
qa_cursor_step(run_id)   # repeat until finished=true
```

Each step returns `visual_digest` from Cursor vision + blank-canvas heuristics.

**Autonomous (background):**

```
qa_start_exploration(profile="full", max_steps=45)
qa_get_run_status(run_id)
```

Use MCP prompt `qa-cursor-driver` for full instructions.

```bash
PYTHONPATH=. .venv/bin/python3 -m qa_agent.mcp_server
```

## Exploration profiles

| Profile | Scope |
|---------|--------|
| `full` | Full playground: locale, datasets, training, replay (default desktop) |
| `playground` | Same as `full` |
| `training` | Training flow: datasets, activations, Play/Step/Reset, replay |
| `i18n` | Locale switcher (en / ru) |
| `mobile_full` | Full playground on mobile viewport |

MCP: `qa_list_profiles` for details.

Coverage tracks **interactions** (not multi-route navigation): locale, dataset/activation picks, hyperparameters, training controls, replay scrubber, canvas.

## MCP tools

| Tool | Description |
|------|-------------|
| `qa_cursor_start` | Start stepwise session; returns `visual_digest` + DOM |
| `qa_cursor_step` | Auto step: Cursor vision picks action, returns next observation |
| `qa_cursor_act` | Execute one action (omit `tool` for auto choice) |
| `qa_start_exploration` | Background autonomous run via Cursor sampling |
| `qa_cursor_abort` | Close browser without report |
| `qa_list_profiles` | List exploration profiles |
| `qa_list_mobile_cursor_slices` | Mobile-focused slices (0–2) |
| `qa_cursor_start_mobile_slice` | Start one mobile slice |
| `qa_list_dataset_slices` | Dataset/activation audit slices (0–2) |
| `qa_cursor_start_dataset_slice` | Start one dataset slice |
| `qa_get_run_status` | Poll run progress |
| `qa_list_runs` | List session + disk history |
| `qa_get_report` / `qa_get_metrics` | Read outputs |
| `qa_compare_runs` | Diff two runs |

## Output

Each run creates a timestamped directory under `qa_agent/runs/`:

- `report.md` — summary, findings, UX metrics, trace
- `metrics.json` — machine-readable UX scores
- `events.jsonl` — structured event log
- `screenshots/step-XX.png` — per-step screenshots

## Tests

```bash
PYTHONPATH=. .venv/bin/pytest qa_agent/test_*.py -q
```

## Notes

- Runs are exploratory and non-deterministic; keep unit tests in `packages/core` as CI source of truth.
- No login, API seed, or backend required — static SPA only.
