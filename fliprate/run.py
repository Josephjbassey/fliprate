"""Run the rebuttal protocol for one model; resumable JSONL logging.

Output: <out-dir>/<model-slug>.jsonl (one line per completed item). Re-running skips done ids.
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from .client import ChatClient, OLLAMA_URL
from .protocol import run_item


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def load_items(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for l in f:
            try:
                ids.add(json.loads(l)["id"])
            except Exception:  # noqa: BLE001  (truncated last line)
                pass
    return ids


def run(client: ChatClient, items: list[dict], out_path: Path, seed: int = 0, limit: int | None = None,
        label: str | None = None) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{client.model}: {len(done)} done, {len(todo)} to run -> {out_path}", flush=True)
    n = 0
    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for it in todo:
            t = time.time()
            rec = run_item(client, it, seed=seed)
            rec["base_url"] = client.base_url
            rec["temperature"] = client.temperature
            if label:
                rec["label"] = label
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            n += 1
            print(f"[{n}/{len(todo)}] {it['id']} cond={rec['condition']} {time.time()-t:.1f}s", flush=True)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=OLLAMA_URL, help="Ollama http://localhost:11434/v1 | LM Studio http://localhost:1234/v1")
    ap.add_argument("--model", help="model name as served (see --list-models)")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--out-dir", default="results/runs")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only run first N unfinished items")
    a = ap.parse_args(argv)
    c = ChatClient(a.base_url, a.model or "", temperature=a.temperature, max_tokens=a.max_tokens, seed=a.seed)
    if a.list_models:
        print("\n".join(c.list_models())); return
    if not a.model:
        ap.error("--model is required")
    run(c, load_items(a.questions), Path(a.out_dir) / f"{slug(a.model)}.jsonl", seed=a.seed, limit=a.limit)


if __name__ == "__main__":
    main()
