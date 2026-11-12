"""Build a fixed-seed question set from MMLU (cais/mmlu, config 'all', split 'test').

Uses the Hugging Face datasets-server REST API (requests only; no `datasets` install needed).
Writes data/questions.jsonl and data/questions_meta.json (source, citation, license, seed).
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from datetime import datetime, timezone
from pathlib import Path
import requests

API = "https://datasets-server.huggingface.co/rows"
DATASET, CONFIG, SPLIT = "cais/mmlu", "all", "test"
CITATION = ("Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). "
            "Measuring Massive Multitask Language Understanding. ICLR 2021. arXiv:2009.03300. "
            "Also: Hendrycks et al. (2021). Aligning AI With Shared Human Values. ICLR 2021.")


def fetch_page(offset: int, length: int = 100, retries: int = 5) -> dict:
    for i in range(retries):
        try:
            r = requests.get(API, params={"dataset": DATASET, "config": CONFIG, "split": SPLIT,
                                          "offset": offset, "length": length}, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args(argv)

    first = fetch_page(0, 1)
    total = first["num_rows_total"]
    rng = random.Random(a.seed)
    idxs = sorted(rng.sample(range(total), a.n))
    wanted = set(idxs)
    pages = sorted({i // 100 for i in idxs})
    items = {}
    for p in pages:
        data = fetch_page(p * 100, 100)
        for row in data["rows"]:
            if row["row_idx"] in wanted:
                r = row["row"]
                items[row["row_idx"]] = {
                    "id": f"mmlu-{row['row_idx']}", "source_index": row["row_idx"],
                    "subject": r["subject"], "question": r["question"], "choices": r["choices"],
                    "answer": "ABCD"[r["answer"]]}
        print(f"fetched page {p} ({len(items)}/{a.n})", flush=True)
    assert len(items) == a.n, f"got {len(items)} of {a.n}"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    qpath = out / "questions.jsonl"
    with open(qpath, "w", encoding="utf-8", newline="\n") as f:
        for i in idxs:
            f.write(json.dumps(items[i], ensure_ascii=False) + "\n")
    meta = {"source": f"https://huggingface.co/datasets/{DATASET}", "dataset": DATASET, "config": CONFIG,
            "split": SPLIT, "total_rows_in_split": total, "n_items": a.n, "sampling": "uniform random without replacement",
            "seed": a.seed, "license": "MIT (as listed on the cais/mmlu Hugging Face dataset card)",
            "citation": CITATION, "built_utc": datetime.now(timezone.utc).isoformat(),
            "questions_sha256": hashlib.sha256(qpath.read_bytes()).hexdigest()}
    (out / "questions_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {qpath} and metadata")


if __name__ == "__main__":
    main()
