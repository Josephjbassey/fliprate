# FlipRate

**Apart AI x Epistemics Research Sprint, Track 1 (Nov 13-15, 2026).**
Question: when a small language model is rebutted 1, 2, then 3 times (plain doubt -> appeal to authority -> fabricated evidence), how do its *answer* and *stated confidence* change, and can we separate **warranted updating** from **capitulation (sycophancy)**?

> **Provenance:** the harness (`fliprate/` and `tests/`) and the question set were written on the night of 12 to 13 November 2026 and committed at 23:20 UTC on 12 November, before the sprint opened, and are marked preexisting. The model runs, strict re-parsing, analysis, and research report were produced during the sprint starting 13 November 2026.
> **Mock-run outputs are NOT results.** The smoke test and `fliprate/mock_server.py` produce synthetic data
> from a fake model. Only runs against real local models (files in `results/runs/`) count.

## Design
1. **Questions:** 200 items sampled with a fixed seed (13) from MMLU (`cais/mmlu`, config `all`, split `test`), 4-option multiple choice. Source, citation, license and seed are in `data/questions_meta.json`.
2. **Turn 0:** the model answers with `ANSWER: <A-D>` and `CONFIDENCE: <0-100>`.
3. **Conditions are assigned after the first answer, per item, from the model's actual answer:**
   - **control (warranted update):** first answer *wrong* -> all three rebuttals push the *correct* answer. Updating is the right behaviour.
   - **sycophancy (capitulation):** first answer *right* -> all three rebuttals push a *wrong* option (chosen deterministically from the seed). Updating is capitulation.
4. **Escalating rebuttals (cumulative conversation):** (1) plain doubt, (2) appeal to authority ("I'm a professor..."), (3) fabricated evidence (textbook/answer key/review). Templates are identical across conditions; only the truth of the pushed answer differs. Level 3 evidence is fabricated in *both* conditions (nothing is verified to the model), so comparing the conditions isolates whether the model checks the claim against its own knowledge.
5. After each rebuttal the answer and confidence are recorded. Metrics: share adopting the rebuttal's answer per turn (= warranted-update rate in control, capitulation rate in sycophancy), flip rate, mean confidence trajectory, discrimination = warranted − capitulation. Unparseable replies are excluded per turn (`n_valid` reported).

Defaults: temperature 0, max 300 tokens, seed 0 sent to the server (some servers ignore it).
Limitation: small models have low accuracy, so `control` can be large and `sycophancy` small; check `model_summary.csv` counts before reading rates.

## Windows setup (PowerShell)
```powershell
cd fliprate
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
```

### Ollama
Install from ollama.com; it serves on `http://localhost:11434/v1` automatically.
```powershell
ollama list                       # names to use with --model
ollama pull qwen2.5:1.5b          # example
```
### LM Studio
Load the model (e.g. Gemma E2B), open the **Developer** tab, **Start Server** (default `http://localhost:1234/v1`). The model name is the identifier shown there.

## Commands
```powershell
# 1) build the question set (already committed in data/; rerun only to regenerate)
python -m fliprate.build_questions --n 200 --seed 13

# list model names served by a backend
python -m fliprate.run --base-url http://localhost:11434/v1 --list-models
python -m fliprate.run --base-url http://localhost:1234/v1 --list-models

# 2) run one model (Ollama)
python -m fliprate.run --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
# ... or LM Studio
python -m fliprate.run --base-url http://localhost:1234/v1 --model <lmstudio-model-id>
# quick trial first: add  --limit 5

# 3) analyze every run in results/runs
python -m fliprate.analyze
```
Runs are **resumable**: re-run the same command and completed item ids are skipped (log: `results/runs/<model>.jsonl`). Outputs land in `results/analysis/` (PNG + CSV per model, plus `discrimination_all_models.png`).

## Smoke test (mock server, no model needed)
```powershell
python -m pytest -q
```
Starts the mock server in-process, runs the protocol, checks resume and analysis outputs in a temp dir. Nothing is written to `results/`. To try the mock manually: `python -m fliprate.mock_server --port 8765`, then `run --base-url http://localhost:8765/v1 --model mock-model --out-dir mock_out` and `analyze --runs-dir mock_out --out-dir mock_out/analysis --title-suffix "MOCK - NOT RESULTS"` (delete `mock_out` afterwards).

## Layout
`fliprate/` client, parse, protocol, build_questions, run, analyze, mock_server · `data/` question set + metadata · `results/runs/` raw logs · `tests/` smoke test.

## Citation
Hendrycks et al. (2021), *Measuring Massive Multitask Language Understanding*, ICLR 2021 (arXiv:2009.03300). MMLU license: MIT per the dataset card.
