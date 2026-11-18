# Bits Saved

Code and results for **"Bits Saved: Sealed Compression Progress, Document-Mixture Coherence, and Information Gain for Language-Model Agents"** (Uday Kamath\*, Kevin Keenan and Vikram Kamat).

\* Corresponding author

The paper argues that an LLM agent should score a candidate document by the bits it saves when the model re-encodes a *sealed* part of the agent's history — history that was never given to the process that produced the candidate. This repo contains the notebooks that run every experiment and the result files the paper reads its numbers, tables and figures from.

> **Status:** the notebooks were executed on Colab GPUs (FULL mode) by the authors. This repo stores the notebooks and their outputs; nothing here has been re-run.

## Layout

```
notebooks/
  compression_progress_llm_agents_v9.ipynb   # theory checks, exact coders, all synthetic-world LLM experiments
  evalplus_sealed_gain.ipynb                 # real-code check on EvalPlus (HumanEval+ and MBPP+)
scripts/
  inputs_from_notebook.py    # fallback: rebuild paper_inputs/ from a notebook's printed output
  make_replication.py        # build replication.tex from the two earlier runs
  regen_bandit_figures.py    # redraw the exact-coder figure with the paper's legend labels
  evalplus_tasks.py          # build the EvalPlus task file from the EvalPlus release files
  build_evalplus.py          # build numbers_evalplus_extra.tex from the EvalPlus results
data/
  evalplus_tasks.json.gz     # the 266 EvalPlus tasks, built by scripts/evalplus_tasks.py
results/
  synthetic/
    paper_inputs/            # numbers.tex, tables and figures read by the paper (written by the v9 notebook)
    earlier_runs/            # v7_numbers.tex, v8_numbers.tex: two earlier full runs on other world draws
    replication.tex          # headline numbers of those runs, for the Robustness paragraph
  evalplus/
    numbers_evalplus.tex         # written by the EvalPlus notebook
    numbers_evalplus_extra.tex   # built by scripts/build_evalplus.py from the files below
    results.json                 # EvalPlus run with the 1.5B observers
    results_7b.json              # EvalPlus run with the 7B observers
    scores_<model>.csv           # per-document scores, one file per observer
requirements.txt
.gitignore
```

The scripts lived in a folder called `tools/` in the authors' working copy, so their docstrings and the header comments of the files they generated say `tools/...`; here they are in `scripts/`.

## Where each result comes from

| Paper element | File | Produced by |
|---|---|---|
| All synthetic-world numbers (`\num{...}`) | `results/synthetic/paper_inputs/numbers.tex` | v9 notebook, Part 4 |
| Table 1 (ranking fidelity) | `paper_inputs/tab_e1.tex` | v9 notebook, Part 4 |
| Table 2 (reading agents) | `paper_inputs/tab_agent.tex` | v9 notebook, Part 4 |
| Table 3 (observers) | `paper_inputs/tab_observers.tex` | v9 notebook, Part 4 |
| Not in the paper (per-type and ablation tables) | `paper_inputs/tab_types.tex`, `tab_ablation.tex` | v9 notebook, Part 4 |
| Figure 2 (exact-coder allocation) | `paper_inputs/fig1b_toy_allocation.pdf` | v9 notebook, Part 2; legend relabelled by `scripts/regen_bandit_figures.py` |
| Figure 3 (signal × document-type heatmap) | `paper_inputs/fig3_e1_heatmap.pdf` | v9 notebook, cell 3.6 |
| Figure 4 (CP₁ vs audit length) | `paper_inputs/fig13_e8b_breakeven.pdf` | v9 notebook, cell 3.8b |
| "Robustness across world draws" | `results/synthetic/replication.tex` | `scripts/make_replication.py` from `earlier_runs/` |
| "Real code" paragraph | `results/evalplus/numbers_evalplus*.tex` | EvalPlus notebook; `_extra` by `scripts/build_evalplus.py` |
| Consistency of prose with results | `paper_inputs/prose_checks.txt` | v9 notebook (42/42 checks passed) |

In the EvalPlus numbers files, the key prefixes are the four observers: `ep` = Qwen2.5-1.5B-Instruct (the main observer), `ep2` = Qwen2.5-Coder-1.5B-Instruct, `ep3` = Qwen2.5-7B-Instruct, `ep4` = Qwen2.5-Coder-7B-Instruct.

`scripts/inputs_from_notebook.py` is kept for the record: it rebuilt `paper_inputs/` from the printed output of the earlier v7 notebook (not included) when that run's results folder was unavailable. The paper's current `paper_inputs/` was written directly by the v9 notebook and does not depend on it.

## Running

Both notebooks are written for Google Colab with a GPU (T4 or better).

**Main notebook.** Set `MODE` in cell 0.2: `"TEST"` (CPU smoke test with a stand-in scorer), `"FAST"` or `"FULL"` (the paper's run). `world_set = 0` gives the reported worlds; `1, 2, …` draw independent sets for replication. Outputs go to `cp_results/`; copy `cp_results/paper_inputs/` to `results/synthetic/paper_inputs/`.

**EvalPlus notebook.** Upload `data/evalplus_tasks.json.gz` when the notebook asks for it. Set `MODE = "FULL"` and run all; outputs go to `evalplus_results/` and are downloaded as `evalplus_results.zip`. The paper uses two runs:

- the 1.5B pair (`Qwen2.5-1.5B-Instruct`, the main observer, and `Qwen2.5-Coder-1.5B-Instruct`) → `results.json`;
- the 7B pair (`Qwen2.5-7B-Instruct`, `Qwen2.5-Coder-7B-Instruct`) → `results_7b.json`.

The committed notebook is configured for the 7B pair; edit `CFG["models"]` in cell 0.1 for the 1.5B run. HumanEval and MBPP are public, so the observers may have seen them in pretraining; renaming the function, stripping docstrings and scoring on EvalPlus's added tests reduce, but do not remove, that risk.

**CPU-only.** Parts 1–2 of the main notebook (exact coders) need no GPU. `scripts/regen_bandit_figures.py` runs them directly. Run it from the repo root: it writes `out/` and `check.json` into the current folder (both gitignored). On Windows, set `$env:PYTHONUTF8 = "1"` first.

```bash
python scripts/regen_bandit_figures.py notebooks/compression_progress_llm_agents_v9.ipynb
```

It selects notebook cells by position and checks exact legend strings, so it works only with the v9 notebook as committed.

**Rebuilding the replication file** (CPU, from the repo root):

```bash
mkdir -p out
python scripts/make_replication.py results/synthetic/earlier_runs/v7_numbers.tex results/synthetic/earlier_runs/v8_numbers.tex > out/replication_check.tex
```

On Windows PowerShell, replace `> out/replication_check.tex` with `| Out-File -Encoding utf8 out/replication_check.tex`. Compare with `results/synthetic/replication.tex`; only the comment line naming the input paths should differ.

**Rebuilding the EvalPlus numbers file** (CPU, needs `results.json`, `results_7b.json` and the four score CSVs). The script overwrites files in the folder it is given, so run it on a copy:

```bash
cp -r results/evalplus out/evalplus_check
python scripts/build_evalplus.py out/evalplus_check
```

Compare `out/evalplus_check/numbers_evalplus_extra.tex` with `results/evalplus/numbers_evalplus_extra.tex`; they should be identical. The script also writes `tab_evalplus_rho.tex`, which the paper does not use.

## Regenerating the EvalPlus task file (optional)

`data/evalplus_tasks.json.gz` holds the 266 tasks with their observations and candidate documents. It is committed, so this step is only needed to verify it. `scripts/evalplus_tasks.py` builds it from the public EvalPlus release files (`humanevalplus_release` at commit `200defc`, `mbppplus_release` at `dadf43d`, the latest as of October 2026). This needs no GPU, takes Python 3.9 or later and `numpy`, and runs on macOS or Linux only (the script uses `signal.SIGALRM`; on Windows, use WSL). It takes about 5 minutes.

The script executes the EvalPlus reference solutions and mutated copies of them, so run it in a separate virtual environment.

**1. Get the EvalPlus release files.** From the folder that *contains* `bits-saved-compression/` (so the clones stay outside the repo):

```bash
git clone https://github.com/evalplus/humanevalplus_release
git clone https://github.com/evalplus/mbppplus_release
```

**2. Set up an environment and build the file** (the script writes to the current folder, so run it from `data/`):

```bash
cd bits-saved-compression
python3 -m venv .venv && source .venv/bin/activate
pip install numpy
cd data
python ../scripts/evalplus_tasks.py ../../humanevalplus_release/HumanEvalPlus.jsonl.gz ../../mbppplus_release/MbppPlus.jsonl.gz
```

It should end with `266 tasks written to evalplus_tasks.json.gz`.

**3. Check the result.** Still in `data/`:

```bash
python -c "import gzip, json; t = json.load(gzip.open('evalplus_tasks.json.gz', 'rt')); he = sum(x['task_id'].startswith('HumanEval') for x in t); print(len(t), 'tasks:', he, 'HumanEval+,', len(t) - he, 'MBPP+')"
```

The runs in the paper used **266 tasks: 109 HumanEval+, 157 MBPP+** (of 542 problems). The authors checked that the script reproduces their file exactly. It is seeded, but it drops calls that take over 1 s and does not pin the EvalPlus release version, so a different count means the rebuild has drifted from the original.

## Not included

- The raw `cp_results/` folder of the v9 run (`results.json`, per-document score CSVs, the full figure set). Every number the paper prints is in `results/synthetic/paper_inputs/`.

## License

The code and results in this repository are released under the MIT License (see `LICENSE`). `data/evalplus_tasks.json.gz` is derived from the EvalPlus releases ([HumanEval+](https://github.com/evalplus/humanevalplus_release), [MBPP+](https://github.com/evalplus/mbppplus_release)), which are licensed under the Apache License 2.0; that file remains subject to their license.

## Requirements

See `requirements.txt`. Colab provides most of these; the main notebook installs `peft` and `accelerate` itself.

## Citation

Kamath, U.; Keenan, K.; Kamat, V. Bits Saved: Sealed Compression Progress, Document-Mixture Coherence, and Information Gain for Language-Model Agents.