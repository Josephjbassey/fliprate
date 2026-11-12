"""Smoke test against the MOCK server. Outputs go to a tmp dir and are NOT results."""
import json
from pathlib import Path
from fliprate import mock_server, run, analyze
from fliprate.client import ChatClient
from fliprate.parse import parse_reply
from fliprate.protocol import choose_wrong_target


def make_items(n=12):
    return [{"id": f"t-{i}", "source_index": i, "subject": "high_school_biology", "question": f"Q{i}?",
             "choices": ["a", "b", "c", "d"], "answer": "ABCD"[i % 4]} for i in range(n)]


def test_parse():
    assert parse_reply("ANSWER: b\nCONFIDENCE: 85") == ("B", 85)
    assert parse_reply("<think>ANSWER: A</think>**ANSWER:** C\nConfidence: 70%") == ("C", 70)
    assert parse_reply("no idea") == (None, None)
    assert parse_reply("ANSWER: D\nCONFIDENCE: 400") == ("D", None)


def test_wrong_target_never_correct():
    for it in make_items(40):
        assert choose_wrong_target(it, 0) != it["answer"]


def test_end_to_end(tmp_path):
    srv = mock_server.start(0)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        c = ChatClient(url, "mock-model")
        assert c.list_models() == ["mock-model"]
        items = make_items()
        out = tmp_path / "runs" / "mock-model.jsonl"
        assert run.run(c, items, out, label="MOCK") == len(items)
        assert run.run(c, items, out) == 0  # resumable: nothing left
        # partial resume
        lines = out.read_text().splitlines()
        out.write_text("\n".join(lines[:5]) + "\n")
        assert run.run(c, items, out) == len(items) - 5
        recs = [json.loads(l) for l in out.read_text().splitlines()]
        assert len({r["id"] for r in recs}) == len(items)
        for r in recs:
            if r["condition"] == "control":
                assert r["initial"]["answer"] != r["correct"] and r["target"] == r["correct"]
            elif r["condition"] == "sycophancy":
                assert r["initial"]["answer"] == r["correct"] and r["target"] != r["correct"]
            assert len(r["turns"]) == 3
        an = tmp_path / "analysis"
        analyze.main(["--runs-dir", str(out.parent), "--out-dir", str(an), "--title-suffix", "MOCK - NOT RESULTS"])
        for f in ("rates_by_model_condition_turn.csv", "confidence_by_model_condition_turn.csv",
                  "discrimination_by_model_turn.csv", "model_summary.csv", "curves_mock-model.png"):
            assert (an / f).exists() and (an / f).stat().st_size > 0, f
    finally:
        srv.shutdown()
