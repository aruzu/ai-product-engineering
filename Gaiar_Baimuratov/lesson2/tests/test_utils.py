"""Unit tests for utility functions in userboard4-baimuratov.py."""

import csv
import json
import importlib.util
from pathlib import Path

# Dynamically load the target module from its file path
ROOT = Path(__file__).resolve().parent.parent
script_path = ROOT / "userboard4-baimuratov.py"

spec = importlib.util.spec_from_file_location("interview", script_path)
interview = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(interview)  # type: ignore


def test_build_description():
    row = {
        "age": "30",
        "role": "Product manager",
        "context": "Works remotely",
        "goals": "Increase team velocity",
        "pain_points": "Context switching",
        "tech_savvy": "High",
    }
    desc = interview._build_description(row)
    assert "30-year" in desc
    assert "Product manager" in desc
    assert "Works remotely" in desc


def test_load_personas_from_csv(tmp_path: Path):
    csv_path = tmp_path / "personas.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "age",
                "role",
                "context",
                "goals",
                "pain_points",
                "tech_savvy",
                "color",
                "emoji",
            ]
        )
        writer.writerow(
            [
                "Dana (Designer)",
                "28",
                "Designer",
                "Works at a startup",
                "Improve user experience",
                "Limited research budget",
                "Medium",
                "cyan",
                "🎨",
            ]
        )

    personas = interview.load_personas_from_csv(str(csv_path))
    assert len(personas) == 1
    persona = personas[0]
    assert persona["name"] == "Dana (Designer)"
    assert "28-year" in persona["description"]
    assert persona["color"] == "cyan"
    assert persona["emoji"] == "🎨"


def test_load_interview_config(tmp_path: Path):
    config_path = tmp_path / "config.json"
    data = {
        "topic": "New SaaS platform for task automation",
        "core_questions": [
            "What problems do you face with current tools?",
            "How do you measure success?",
        ],
        "max_followups": 2,
    }
    config_path.write_text(json.dumps(data), encoding="utf-8")

    cfg = interview.load_interview_config(str(config_path))
    assert cfg["topic"] == data["topic"]
    assert cfg["core_questions"] == data["core_questions"]
    assert cfg["max_followups"] == 2
