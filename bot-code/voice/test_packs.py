"""Offline tests for VoicePack + ahead-of-time flavor (no network)."""
from __future__ import annotations

import json
from pathlib import Path

import flavor
import narrator
import packs
from from_plan import events_from_plan, narrate


def test_pack_switch_changes_lines() -> None:
    print("[1/4] pack switch...")
    packs.set_pack("neutral")
    flavor.clear_script()
    n = narrator.line("pick.grasp", {"id": 3})
    packs.set_pack("boxing")
    flavor.clear_script()
    b = narrator.line("pick.grasp", {"id": 3})
    assert n == "Closing the J7 gripper.", n
    assert b == "Got it — claws closed.", b
    assert n != b
    packs.set_pack("boxing")
    print("  ok")


def test_template_prewrite() -> None:
    print("[2/4] template prewrite...")
    plan = json.loads((Path(__file__).parent / "fixture_plan.json").read_text())
    events = events_from_plan(plan)
    script = flavor.prewrite(events, pack="boxing", flavor="template")
    assert script
    assert any(
        "claw" in v.lower() or "box" in v.lower() or "layer" in v.lower()
        or "scanning" in v.lower() or "planted" in v.lower()
        for v in script.values()), script
    # narrator should prefer script overrides
    text = narrator.line("plan.ready", {"n": 2})
    assert text == script[flavor.script_key("plan.ready", {"n": 2})]
    print(f"  {len(script)} lines ok")


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
        events, pack="neutral", flavor="openai",
        client_factory=lambda: Boom())
    assert script  # templates remain
    assert flavor.lookup("pick.grasp", {"id": 3}) or flavor.lookup("pick.grasp", {"id": 1})
    print("  ok")


def test_openai_rewrite_injected() -> None:
    print("[4/4] openai rewrite (fake client)...")

    class FakeResp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

    class Fake:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    # echo fallbacks with a marker
                    import json as _json
                    items = _json.loads(kwargs["messages"][1]["content"])["items"]
                    lines = [{"key": it["key"], "text": "Hype " + " ".join(it["fallback"].split()[:7])}
                             for it in items]
                    return FakeResp(_json.dumps({"lines": lines}))

    plan = json.loads((Path(__file__).parent / "fixture_plan.json").read_text())
    events = events_from_plan(plan)
    script = flavor.prewrite(
        events, pack="boxing", flavor="openai",
        client_factory=lambda: Fake())
    assert any(v.startswith("Hype ") for v in script.values()), script
    print("  ok")


def main() -> None:
    print("=== voice packs / flavor ===")
    test_pack_switch_changes_lines()
    test_template_prewrite()
    test_openai_fail_open()
    test_openai_rewrite_injected()
    print("PASS")


if __name__ == "__main__":
    main()
