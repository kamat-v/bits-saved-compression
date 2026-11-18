"""Paper inputs for the EvalPlus check, built from the notebook's results.json and per-document score CSVs.

Usage: python3 tools/build_evalplus.py evalplus_inputs
Writes evalplus_inputs/numbers_evalplus_extra.tex and evalplus_inputs/tab_evalplus_rho.tex.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats

d = sys.argv[1]
R = json.load(open(os.path.join(d, "results.json")))
if os.path.exists(os.path.join(d, "results_7b.json")):
    R.update(json.load(open(os.path.join(d, "results_7b.json"))))
MAIN, CODER = "Qwen2.5-1.5B-Instruct", "Qwen2.5-Coder-1.5B-Instruct"
BIG, BIGCODER = "Qwen2.5-7B-Instruct", "Qwen2.5-Coder-7B-Instruct"
MODELS = [(m, t) for m, t in [(MAIN, "ep"), (CODER, "ep2"), (BIG, "ep3"), (BIGCODER, "ep4")] if m in R]
LAB = [("G", "Sealed gain $G$$^\\dagger$"), ("ratio", "Gain per bit $G/S$$^\\dagger$"), ("G_un", "Unsealed gain"),
       ("low_ppl", "Low perplexity"), ("surprise", "Surprise"), ("relevance", "Emb.\\ relevance"), ("judge", "LLM judge"),
       ("random", "Random")]


def num(v, p=2):
    v = 0.0 if round(v, p) == 0 else v
    return "\\ensuremath{-}" + f"{-v:.{p}f}" if v < 0 else f"{v:.{p}f}"


def pct(v):
    return f"{100 * v:.0f}\\%"


def rho_cell(m):
    return f"{num(m[0])} [{num(m[1])}, {num(m[2])}]"


HEAD = {MAIN: "1.5B", CODER: "Coder 1.5B", BIG: "7B", BIGCODER: "Coder 7B"}
rows = [f"{lab} & " + " & ".join(rho_cell(R[m]['signals'][s]['rho']) for m, _ in MODELS) + "\\\\" for s, lab in LAB]
with open(os.path.join(d, "tab_evalplus_rho.tex"), "w") as f:
    f.write("% Built by tools/build_evalplus.py from results.json (and results_7b.json).\n\\begin{tabular}{@{}l" + "c" * len(MODELS) + "@{}}\n\\toprule\n"
            "Signal & " + " & ".join(HEAD[m] for m, _ in MODELS) + "\\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n")

out = ["% Built by tools/build_evalplus.py from results.json and the per-document score CSVs."]
for model, tag in MODELS:
    df = pd.read_csv(os.path.join(d, f"scores_{model}.csv"))
    df["src"] = df.task.str.split("/").str[0]
    for src, key in [("HumanEval", "he"), ("Mbpp", "mbpp")]:
        g = df[df.src == src]
        for s in ["G", "G_un"]:
            r = [stats.spearmanr(t[s], t.U).correlation for _, t in g.groupby("task") if t[s].std() > 0 and t.U.std() > 0]
            out.append(f"\\setnum{{{tag}.{key}.rho.{s.replace('_', '')}}}{{{num(float(np.mean(r)))}}}")
        out.append(f"\\setnum{{{tag}.{key}.n}}{{{g.task.nunique()}}}")
    top = lambda s: df.loc[df.groupby("task")[s].idxmax(), "type"]
    out.append(f"\\setnum{{{tag}.sur.noisetop}}{{{pct((top('surprise') == 'NOISE').mean())}}}")
    out.append(f"\\setnum{{{tag}.un.echotop}}{{{pct((top('G_un') == 'ECHO').mean())}}}")
    out.append(f"\\setnum{{{tag}.G.examplestop}}{{{pct((top('G') == 'EXAMPLES').mean())}}}")
    out.append(f"\\setnum{{{tag}.U.examplestop}}{{{pct((top('U') == 'EXAMPLES').mean())}}}")
    ranks = lambda s: np.mean([t[s].rank(ascending=False)[t.type == "ECHO"].iloc[0] for _, t in df.groupby("task")])
    out.append(f"\\setnum{{{tag}.echo.rank.G}}{{{ranks('G'):.1f}}}")
    out.append(f"\\setnum{{{tag}.echo.rank.Gun}}{{{ranks('G_un'):.1f}}}")
    out.append(f"\\setnum{{{tag}.ncand}}{{{df.groupby('task').size().iloc[0]}}}")
    sig = R[model]["signals"]
    out.append(f"\\setnum{{{tag}.dose.U}}{{{pct(sig['U']['dose'])}}}")
    out.append(f"\\setnum{{{tag}.wrongbelow.U}}{{{pct(sig['U']['wrong_below'])}}}")
    mant, ex = f"{R[model]['p_vs_best']:.1e}".split("e")
    out.append(f"\\setnum{{{tag}.p}}{{{mant}\\times 10^{{{int(ex)}}}}}")
    out.append(f"\\setnum{{{tag}.wrongbelow.G}}{{{pct(sig['G']['wrong_below'])}}}")
    others = [s for s in ["G_un", "low_ppl", "surprise", "relevance", "judge"]]
    out.append(f"\\setnum{{{tag}.model}}{{{model}}}")
    out.append(f"\\setnum{{{tag}.rho.G}}{{{num(sig['G']['rho'][0])}}}")
    out.append(f"\\setnum{{{tag}.rho.Gun}}{{{num(sig['G_un']['rho'][0])}}}")
    out.append(f"\\setnum{{{tag}.cp1pos}}{{{pct(R[model]['cp1_pos_useful'])}}}")
    out.append(f"\\setnum{{{tag}.rho.othersmax}}{{{num(max(sig[s]['rho'][0] for s in others if s != 'G_un'))}}}")
open(os.path.join(d, "numbers_evalplus_extra.tex"), "w").write("\n".join(out) + "\n")
print("written")
