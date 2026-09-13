"""Offline tests for VoicePack + ahead-of-time flavor (no network)."""
from __future__ import annotations

import json
from pathlib import Path

import flavor
import narrator
import packs
from from_plan import events_from_plan


def test_pack_switch() -> None:
    print("[1/4] pack switch...")
    packs.set_pack("neutral")
    flavor.clear_script()
    n = narrator.line("pick.grasp", {"id": 3})
    packs.set_pack("trump")
    flavor.clear_script()
    # No hand-written trump templates → falls back to neutral telemetry text
    t = narrator.line("pick.grasp", {"id": 3})
    assert n == "Closing the J7 gripper.", n
    assert t == n  # bland fail-open until OpenAI fills script
    assert "trump" in packs.list_packs()
    assert "Trump" in packs.get_pack("trump").style
    print("  ok")


def test_template_prewrite_failopen() -> None:
    print("[2/4] template prewrite (fail-open)...")
    plan = json.loads((Path(__file__).parent / "fixture_plan.json").read_text())
    events = events_from_plan(plan)
    script = flavor.prewrite(events, pack="trump", flavor="template")
    assert script
    # Without OpenAI, trump uses narrator.NEUTRAL fallthrough — not character lines
    grasp_key = next((k for k in script if k.startswith("pick.grasp")), None)
    assert grasp_key and "gripper" in script[grasp_key].lower(), script
    print(f"  {len(script)} fail-open lines ok")


def test_openai_fail_open() -> None:
    print("[3/4] openai fail-open...")

    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**_):
                    raise RuntimeError("no network")

    plan = json.loads((Path(__file__).parent / "fixture_plan.json").read_text())
    events = events_from_plan(plan)
    script = flavor.prewrite(
        events, pack="trump", flavor="openai",
        client_factory=lambda: Boom())
    assert script
    print("  ok")


def test_openai_prompt_generates() -> None:
    print("[4/4] openai prompt (fake client)...")

    class FakeResp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

    class Fake:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    import json as _json
                    body = _json.loads(kwargs["messages"][1]["content"])
                    assert "persona" in body
                    assert "We're gonna build a big beautiful wall" in body["persona"]
                    items = body["items"]
                    assert any(it["key"] == "startup" for it in items)
                    lines = [{"key": it["key"],
                              "text": f"Tremendous {it['event']} folks"}
                             for it in items]
                    return FakeResp(_json.dumps({"lines": lines}))

    plan = json.loads((Path(__file__).parent / "fixture_plan.json").read_text())
    events = events_from_plan(plan)
    script = flavor.prewrite(
        events, pack="trump", flavor="openai",
        client_factory=lambda: Fake())
    assert script.get("startup", "").startswith("Tremendous")
    assert any(v.startswith("Tremendous") for v in script.values())
    print("  ok")


def main() -> None:
    print("=== voice packs / flavor ===")
    test_pack_switch()
    test_template_prewrite_failopen()
    test_openai_fail_open()
    test_openai_prompt_generates()
    print("PASS")


if __name__ == "__main__":
    main()
