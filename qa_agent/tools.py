from __future__ import annotations

from typing import Any

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate to a relative or absolute URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Target URL, e.g. /",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an interactive element by ref from the DOM digest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element ref such as e1, e2",
                    }
                },
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill",
            "description": "Fill an input or textarea by ref.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["ref", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_testid",
            "description": "Click an element by data-testid when present in the DOM digest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "testid": {"type": "string"},
                },
                "required": ["testid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_option",
            "description": "Select an option on a <select> element by ref from the DOM digest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "value": {
                        "type": "string",
                        "description": "Option value attribute to select",
                    },
                },
                "required": ["ref", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_testid",
            "description": "Select an option on a select[data-testid=...] element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "testid": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["testid", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a keyboard key, e.g. Enter or Escape.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the page up or down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Scroll amount in pixels",
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_back",
            "description": "Go back in browser history.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_appearance",
            "description": (
                "Instantly apply theme mode and/or color scheme (localStorage + html data attrs). "
                "Schemes: study, classic, sepia, neon, ethereal. "
                "Modes: light, dark."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "theme_mode": {
                        "type": "string",
                        "enum": ["light", "dark"],
                        "description": "Resolved theme mode to apply",
                    },
                    "color_scheme": {
                        "type": "string",
                        "enum": [
                            "study",
                            "classic",
                            "sepia",
                            "neon",
                            "ethereal",
                        ],
                        "description": "Palette id applied app-wide",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for",
            "description": "Wait for UI/network activity to settle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_ux_observation",
            "description": (
                "Rate a UX aspect of the current screen on a 1-5 scale "
                "(1=poor, 5=excellent). Use for clarity, findability, feedback, "
                "consistency, and efficiency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "UI area being rated, e.g. login form",
                    },
                    "aspect": {
                        "type": "string",
                        "enum": [
                            "clarity",
                            "findability",
                            "feedback",
                            "consistency",
                            "efficiency",
                        ],
                    },
                    "score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    },
                    "notes": {"type": "string"},
                },
                "required": ["area", "aspect", "score"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_issue",
            "description": "Record a bug or UX issue found during exploration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "major", "minor", "cosmetic"],
                    },
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "repro_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["severity", "title", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish exploration and provide a short summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                },
                "required": ["summary"],
            },
        },
    },
]

BROWSER_TOOLS = {
    "navigate",
    "click",
    "click_testid",
    "fill",
    "select_option",
    "select_testid",
    "press_key",
    "scroll",
    "go_back",
    "wait_for",
    "set_appearance",
}
