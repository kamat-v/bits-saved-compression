#!/usr/bin/env python3
"""Build paper_inputs/ from the printed output of an executed notebook (used when cp_results.zip is not at hand).

    python tools/inputs_from_notebook.py executed_notebook.ipynb paper_inputs

Every value is parsed from text the notebook printed, at the precision it printed. Two things the notebook only
shows inside figures are transcribed below (the annotations printed in those figures) and marked HAND-TRANSCRIBED:
the observer cross-correlation matrix and the heatmap of Figure 3. The paper_inputs folder inside cp_results.zip is
the exact version and should replace this one when available."""
import json, re, sys, os
import numpy as np

nb_path, out_dir = sys.argv[1], sys.argv[2]
nb = json.load(open(nb_path)); os.makedirs(out_dir, exist_ok=True)
txt = ""
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    for o in c.get("outputs", []):
        if o["output_type"] == "stream": txt += "".join(o["text"])
        elif "data" in o and "text/plain" in o["data"]: txt += "".join(o["data"]["text/plain"]) + "\n"

# ---- HAND-TRANSCRIBED from the annotated figures of the executed notebook (v7 FULL run, 2026-10-05) ----------------
OBS_NAMES = ["0.5B", "1.5B", "3B", "7B"]
XM = np.array([[0.54, 0.29, 0.33, 0.25], [0.59, 0.62, 0.61, 0.56], [0.56, 0.54, 0.58, 0.59], [0.50, 0.56, 0.56, 0.55]])  # rho(G_A, U_B), cell 3.13 panel (b)
HEAT_COLS = ["U", "CP", "CP_ratio", "IG_full", "surprise", "relevance", "novelty", "judge"]
HEAT_ROWS = ["RULE", "PARTIAL", "INDIRECT", "EXAMPLES", "WRONG", "ECHO", "DUP", "DISTRACTOR", "SHUFFLED", "NOISE", "BOILER"]
HEAT = [[2.0, 2.9, 2.5, 1.8, -0.8, 0.5, -0.4, 0.6], [1.0, 1.1, 1.5, 0.5, -0.7, 0.5, -0.5, -0.2], [0.5, 0.0, 0.0, -0.3, -0.0, 0.3, -0.4, 0.7],
        [0.4, 1.3, 0.7, 1.0, -0.2, 0.6, -0.3, 0.6], [-1.3, -0.4, -0.4, -0.6, -0.7, 0.5, -0.4, 0.2], [0.1, -0.5, -0.6, 1.9, -0.8, 0.6, -0.4, 0.4],
        [0.1, -0.3, 0.1, -0.5, -0.9, 0.3, -0.6, 0.9], [-0.9, -0.7, -0.6, -0.8, -0.1, 0.5, -0.4, -0.9], [-0.6, -0.5, -0.5, -0.7, 1.3, 0.2, -0.2, 1.2],
        [0.3, -0.3, -0.5, -0.6, 1.8, -2.4, 2.5, -0.0], [0.1, -0.6, -0.6, -0.7, 0.4, -0.6, 0.2, -2.0]]      # cell 3.6, first figure
# Not printed by the notebook: largest gap between the sealed and unsealed forms of CP_1 for the coherent coder. Parts 1-2 are
# deterministic CPU code; this value comes from running them locally (the printed table shows the two columns agree).
TOY_COHDIFF = "\\ensuremath{2.0\\times 10^{-13}}"
# ---------------------------------------------------------------------------------------------------------------------

MINUS = "\\ensuremath{-}"
def tex(s):                       # a number string as printed -> LaTeX (proper minus sign)
    s = str(s).strip(); return (MINUS + s[1:]) if s.startswith("-") else s
def fnum(x, d=2, plus=False):
    s = f"{abs(x):.{d}f}"; neg = x < 0 and float(s) != 0
    return (MINUS if neg else ("+" if plus else "")) + s
def pct(x): return f"{100 * float(x):.0f}\\%"
def pval(s):                      # p-values are printed with two significant digits; keep them as printed
    s = s.strip(); m = re.fullmatch(r"([\d.]+)e-0?(\d+)", s)
    return f"\\ensuremath{{{m.group(1)}\\times 10^{{-{int(m.group(2))}}}}}" if m else s
def one(pat, flags=0):
    m = re.search(pat, txt, flags)
    if not m: raise SystemExit("pattern not found: " + pat)
    return m.groups() if m.lastindex and m.lastindex > 1 else m.group(1)
def table(header_pat, n_rows, label_width=None):
    """Rows of a printed pandas table: returns {row label: [cells]} splitting on 2+ spaces."""
    i = re.search(header_pat, txt, re.M).end(); rows = {}
    for ln in txt[i:].split("\n")[1:]:
        if not ln.strip() or ln.startswith(("✔", "✘", "[", "RULE break")): break
        m = re.match(r"(.*?\S)\s{2,}(.*)", ln.strip())          # label, then cells ("0.61 [0.56, 0.66]" counts as one cell)
        if m: rows[m.group(1)] = re.findall(r"-?[\d.]+ \[[^\]]+\]|\S+", m.group(2))
        if len(rows) == n_rows: break
    return rows

NUM = {}
PROSE = {"IG_full": "unsealed gain", "surprise": "surprise", "low_ppl": "low perplexity", "novelty": "embedding novelty",
         "relevance": "embedding relevance", "judge": "the LLM judge", "random": "random choice", "CP_ratio": "gain per bit read"}
ROW = {"Sealed gain $G$ (ours)": "Sealed gain $G$\\,$^\\dagger$", "Gain per bit $G/S$ (ours)": "Gain per bit $G/S$\\,$^\\dagger$", "Unsealed gain": "Unsealed gain",
       "Surprise": "Surprise", "Low perplexity": "Low perplexity", "Embedding novelty": "Emb. novelty", "Embedding relevance": "Emb. relevance",
       "LLM judge": "LLM judge", "Random": "Random", "Oracle (true utility)": "Oracle"}
assert "MODE: FULL" in txt, "this is not a FULL run"

# ---- configuration
n_pairs, n_dev, n_test = one(r"E1 done in \d+s: (\d+) \(world, doc\) pairs; dev worlds=(\d+), test worlds=(\d+)")
lam_dict = {float(a): float(b) for a, b in re.findall(r"([\d.]+): np\.float64\(([\d.]+)\)", one(r"dev Spearman by lambda: (\{.*?\})"))}
lam = float(one(r"chosen lambda = ([\d.]+)")); assert lam == 0.0
NUM.update({"cfg.nworlds": int(n_dev) + int(n_test), "cfg.ndev": n_dev, "cfg.ntest": n_test, "cfg.lam": f"{lam:g}",
            "cfg.nagent": one(r"agent world \d+/(\d+)"), "cfg.steps": one(r"agent_steps: (\d+)"), "cfg.budgetx": one(r"agent_budget_x: ([\d.]+)"),
            "cfg.budget": one(r"token budgets: [\d.]+ x the full guide = (\d+) tokens"), "cfg.nobs": one(r"n_worlds_observer: (\d+)"),
            "cfg.nlong": one(r"at (\d+) audit lines"),
            "dev.rho.lam0": fnum(lam_dict[0.0]), "dev.rho.maxpos": fnum(max(v for k, v in lam_dict.items() if k > 0)), "dev.rho.lam1": fnum(lam_dict[1.0])})

# ---- exact coders
coh = table(r"^Document \(coherent coder\) *$", 5); tch = table(r"Teachable \(incoherent\) coder:\n.*\n^doc *$", 5)
eg, eglo, eghi, ndraws = one(r"E\[G\] = ([\d.]+) \[([\d.]+), ([\d.]+)\] bits over (\d+) prior-predictive draws")
maxcp, npairs = one(r"max CP_1 \(sealed and unsealed forms\) over (\d+) coherent cases = (-?[\d.e+-]+) bits")[::-1]
teach = {k: float(v[6]) for k, v in tch.items()}; best = max(teach, key=teach.get)
assert "RULE" in best
s_n, r_n, i_n, c_n, re_n = (one(rf"time on noise TVs?: {a} ([\d.]+) \[") for a in ("surprise", "retrospective", "surprise", "surprise", "retrospective"))
cp_noise = one(r"time on noise TV: surprise [\d.]+ \[[\d., ]+\] vs CP ([\d.]+)")
toytab = table(r"^Intrinsic reward *$", 6)
ecp, ernd, enov, p_rnd = one(r"final error CP ([\d.]+) vs random ([\d.]+) vs novelty ([\d.]+) \(Wilcoxon vs random p=([\d.e-]+)\)")
esur = one(r"final error CP [\d.]+ vs surprise ([\d.]+) bits/sym")
u_early = [float(x) for x in one(r"first (\d+) steps: \[([\d., ]+)\]")[1].split(",")]; u_steps = one(r"first (\d+) steps: \[")
sur_full = [float(x) for x in one(r"surprise attention \(full run\): \[([\d., ]+)\]").split(",")]
NUM.update({"toy.pairs": f"{int(npairs):,}".replace(",", "{,}"), "toy.maxcp": fnum(float(maxcp)), "toy.teach": pct(teach[best]),
            "toy.teachother": pct(max(v for k, v in teach.items() if k != best)), "toy.cohdiff": TOY_COHDIFF, "toy.eg": eg, "toy.eglo": eglo, "toy.eghi": eghi, "toy.ndraws": ndraws,
            "toy.echofull": one(r"ECHO: G_full=([\d.]+) bits"), "toy.echoseal": one(r"ECHO: G_full=[\d.]+ bits vs G_seal=([\d.]+) bits"),
            "toy.seeds": one(r"toy_seeds: (\d+)"), "toy.steps": one(r"toy_steps: (\d+)"),
            "toy.noise.surprise": pct(s_n), "toy.noise.cp": pct(cp_noise), "toy.noise.retro": pct(r_n),
            "toy.noise.infogain": pct(toytab["Bayesian info-gain (KL)"][1].split(" ")[0]),
            "toy.err.cp": fnum(float(ecp)), "toy.err.random": fnum(float(ernd)), "toy.err.novelty": fnum(float(enov)), "toy.err.surprise": fnum(float(esur)),
            "toy.p.random": pval(p_rnd), "toy.u.early": u_steps, "toy.u.new": pct(u_early[2]), "toy.u.known": pct(u_early[0]), "toy.u.noise": pct(u_early[-1]),
            "toy.u.surpnoise": pct(sur_full[-1])})
assert max(range(6), key=lambda i: u_early[i]) == 2
# sealed and unsealed CP_1 of the coherent coder agree in the printed table (columns CP1 and CP1_seal); the exact gap is in the zip
assert all(v[3] == v[4] for v in coh.values())

# ---- E1
T1 = table(r"^Signal *$", 10)
def t1(lab): c = T1[lab]; return dict(rho=c[0], dacc=c[3], honey=c[5], wrong3=c[6])
cp, orc = t1("Sealed gain $G$ (ours)"), t1("Oracle (true utility)")
rcp, bname, rbest, p_e1 = one(r"Spearman CP=([\d.-]+) vs best baseline \((\w+)\)=([\d.-]+); paired Wilcoxon p=([\d.e-]+)")
NUM.update({"e1.rho.CP": rcp, "e1.rho.best": tex(rbest), "e1.best": PROSE[bname], "e1.p": pval(p_e1), "e1.dacc.CP": tex(cp["dacc"]), "e1.dacc.oracle": tex(orc["dacc"]),
            "e1.honey.CP": pct(cp["honey"]), "e1.wrongbelow": one(r"CP ranks every WRONG rule below RULE in (\d+)% of worlds") + "\\%"})
def ci(s):
    m = re.fullmatch(r"(-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]", s); return f"{tex(m.group(1))} [{tex(m.group(2))}, {tex(m.group(3))}]"
order = sorted([k for k in T1 if not k.startswith("Oracle")], key=lambda k: -float(T1[k][0].split(" ")[0]))
rows = [f"{ROW[k]} & {ci(T1[k][0])} & {tex(T1[k][3])} & {T1[k][5]} \\\\" for k in order]
open(os.path.join(out_dir, "tab_e1.tex"), "w").write("\\begin{tabular}{@{}lccc@{}}\n\\toprule\nSignal & $\\rho$ [95\\% CI] & Acc. & Distr. \\\\\n\\midrule\n" + "\n".join(rows) +
    f"\n\\midrule\nOracle & {T1['Oracle (true utility)'][0].split(' ')[0]} & {tex(orc['dacc'])} & {orc['honey']} \\\\\n\\bottomrule\n\\end{{tabular}}\n")

# ---- E4, E8, E8b, E9, E7
dose = dict(re.findall(r"(\w+)=([\d.]+)", one(r"P\(top-1 useful\) at 12 honeypots: (.*)")))
db = max([k for k in dose if k not in ("CP", "CP_ratio")], key=lambda k: float(dose[k]))
NUM.update({"dose.k": "12", "dose.CP": pct(dose["CP"]), "dose.maxbase": pct(dose[db]), "dose.maxbasename": PROSE[db]})
typ = table(r"^Document type *$", 11)                 # CP1>0 unsealed, CP1>0 sealed, mean S, mean G, mean U
T = lambda t, j, d: fnum(float(typ[t][j]), d)
NUM.update({"typ.rule.S": T("RULE", 2, 0), "typ.rule.G": T("RULE", 3, 1), "typ.rule.U": T("RULE", 4, 1), "typ.noise.S": T("NOISE", 2, 0),
            "typ.noise.G": T("NOISE", 3, 1), "typ.noise.U": T("NOISE", 4, 1), "typ.wrong.G": T("WRONG", 3, 1), "typ.wrong.U": T("WRONG", 4, 1),
            "cert.mean12": fnum(float(typ["RULE"][3]) - float(typ["RULE"][2]), 0),
            "cert.any12": pct(max(float(typ[t][1]) for t in ("RULE", "PARTIAL", "INDIRECT", "EXAMPLES")))})
assert NUM["cert.any12"] == "0\\%"                    # all four useful types print 0.0, so the share over useful documents is 0
eu, es = one(r"mean rank of ECHO docs \(1 = top\): unsealed ([\d.]+) vs sealed CP ([\d.]+)")
NUM.update({"abl.echo.unsealed": eu, "abl.echo.sealed": es, "cert.rate": fnum(float(one(r"the full field guide saves ([\d.]+) audit bits per bit read")))})
fin8 = table(r"^Document \(audit = \d+ lines\) *$", 6)
be = [float(x) for x in one(r"RULE break-even audit length per world: \[(.*?)\]").split(",")]; crossed = sorted(x for x in be if x == x)
n_final = round(float(fin8["RULE"][1]) * len(be))
NUM.update({"cert.n": len(crossed), "cert.share": pct(len(crossed) / len(be)), "cert.nfinal": n_final, "cert.mean": fnum(float(fin8["RULE"][0]), 0),
            "cert.wrong": fnum(float(fin8["WRONG"][0]), 0)})
if crossed: NUM.update({"cert.min": f"{crossed[0]:.0f}", "cert.max": f"{crossed[-1]:.0f}", "cert.median": f"{np.median(crossed):.0f}"})
g9 = table(r"^Quantity *$", 5)
NUM.update({"gaz.rule.with": fnum(float(g9["RULE sealed gain (bits)"][0]), 0), "gaz.rule.without": fnum(float(g9["RULE sealed gain (bits)"][1]), 0),
            "gaz.wrong.with": fnum(float(g9["WRONG sealed gain (bits)"][0]), 0), "gaz.wrong.without": fnum(float(g9["WRONG sealed gain (bits)"][1]), 0),
            "gaz.rho.without": g9["Spearman ρ with U"][1], "gaz.wrong3.without": pct(g9["wrong rule in top-3"][1])})
T7 = table(r"^Method *$", 9)
pk, ppos, prho, pread = one(r"best peek = peek (\d+) tokens \((\w+)\): ρ=([\d.]+) reading (\d+)%")
NUM.update({"peek.k": pk, "peek.pos": "start" if ppos == "head" else "middle", "peek.rho": prho, "peek.read": pread + "\\%",
            "peek.learn": fnum(float(T7["learned predictor (16-token peek)"][1]))})

# ---- E2 / E2b / E3
A1 = table(r"^Agent's signal *$", 9)
i2 = txt.index("E2b: token-budget agents") if "E2b: token-budget agents" in txt else txt.index("token budgets:")
_txt, txt = txt, txt[txt.index("token budgets:"):]; A2 = table(r"^Agent's signal *$", 9); txt = _txt
acc = lambda c: c[0].split(" ")[0]
ours, ratio = "Sealed gain $G$ (ours)", "Gain per bit $G/S$ (ours)"
ev = one(r"(final future accuracy: CP .*)")
a_cp, bacc, a_b, p_acc, b_cp, bbits, b_b, p_bits = re.search(r"CP ([\d.]+) vs best baseline (\w+) ([\d.]+) \(Wilcoxon p=([\d.e-]+)\); future bits/token: CP ([\d.]+) vs best baseline (\w+) ([\d.]+) \(p=([\d.e-]+)\)", ev).groups()
lab_of = {"IG_full": "Unsealed gain", "judge": "LLM judge", "relevance": "Embedding relevance", "novelty": "Embedding novelty", "surprise": "Surprise", "random": "Random"}
NUM.update({"ag.acc.CP": acc(A1[ours]), "ag.bpt.CP": A1[ours][1], "ag.best": PROSE[bbits], "ag.acc.best": acc(A1[lab_of[bbits]]), "ag.bpt.best": A1[lab_of[bbits]][1],
            "ag.p.bpt": pval(p_bits), "ag.acc.ratio": acc(A1[ratio]), "ag.bpt.ratio": A1[ratio][1], "ag.tok.ratio": A1[ratio][4], "ag.tok.CP": A1[ours][4]})
if bacc == bbits: NUM["ag.p.acc"] = pval(p_acc)       # the accuracy test printed by the notebook is against the same baseline
b_ratio, b_cpv, bb, b_best, p_b = one(r"future bits/token: ratio ([\d.]+), uncharged CP ([\d.]+), best baseline (\w+) ([\d.]+) \(Wilcoxon ratio vs baseline p=([\d.e-]+)\)")
NUM.update({"agb.bpt.ratio": b_ratio, "agb.bpt.CP": b_cpv, "agb.best": PROSE[bb], "agb.bpt.best": b_best, "agb.p": pval(p_b)})
order = sorted([k for k in A1 if not k.startswith("Oracle")], key=lambda k: float(A1[k][1])) + ["Oracle (true utility)"]
rows = [("\\midrule\n" if k.startswith("Oracle") else "") + f"{ROW[k].replace(chr(92) + ',$^' + chr(92) + 'dagger$', '')} & {acc(A1[k])} & {A1[k][1]} & {A1[k][4]} & {acc(A2[k])} & {A2[k][1]} & {A2[k][3]} \\\\" for k in order]
open(os.path.join(out_dir, "tab_agent.tex"), "w").write("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{" + f"{NUM['cfg.steps']} reads" +
    "} & \\multicolumn{3}{c}{" + f"Budget {NUM['cfg.budgetx']}$\\times$" + "} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\nSignal & Acc. & b/t & Tok. & Acc. & b/t & Tok. \\\\\n\\midrule\n" +
    "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
rb, ra, pb_, pa_ = one(r"CP of RULE: before ([\d.-]+) bits → after ([\d.-]+) bits; PARTIAL after RULE: ([\d.-]+) → ([\d.-]+)")
NUM.update({"bore.rule.before": tex(rb), "bore.rule.after": tex(ra), "bore.partial.before": tex(pb_), "bore.partial.after": tex(pa_)})   # lambda = 0, so CP = sealed gain

# ---- E5 observers, E6
diag, off = one(r"mean ρ\(CP_A, U_A\)=([\d.]+) vs ρ\(CP_A, U_B≠A\)=([\d.]+)")
ind = re.findall(r"Qwen2\.5-([\d.]+B)=(-?[\d.]+)", one(r"U\(INDIRECT\)/U\(RULE\): (.*)"))
adv = re.findall(r"Qwen2\.5-([\d.]+B): (-?[\d.]+) → (-?[\d.]+) bits \(p=([\d.e-]+)\)", one(r"CP\(ADVANCED\) without → with primer: (.*)"))
assert [a for a, _ in ind] == OBS_NAMES == [a[0] for a in adv]
k = len(OBS_NAMES); cross = [(XM[i].sum() - XM[i, i]) / (k - 1) for i in range(k)]
assert abs(np.mean(np.diag(XM)) - float(diag)) < 0.005 and abs(np.mean(cross) - float(off)) < 0.005, "transcribed matrix disagrees with the printed means"
NUM.update({"obs.self": diag, "obs.cross": off, "obs.k": k, "obs.nselfbest": int(sum(XM[i, i] >= XM[i].max() for i in range(k))),
            "obs.nselfgt": int(sum(XM[i, i] > cross[i] for i in range(k))), "obs.self0": fnum(XM[0, 0]), "obs.cross0": fnum(cross[0]),
            "obs.name.first": OBS_NAMES[0], "obs.name.last": OBS_NAMES[-1], "obs.ind.first": tex(ind[0][1]), "obs.ind.last": tex(ind[-1][1]),
            "edu.before": tex(adv[-1][1]), "edu.after": tex(adv[-1][2]), "edu.p": pval(adv[-1][3])})
rows = [f"{o} & {fnum(XM[i, i])} & {fnum(cross[i])} & {tex(ind[i][1])} & {tex(adv[i][1])} & {tex(adv[i][2])} & {pval(adv[i][3])} \\\\" for i, o in enumerate(OBS_NAMES)]
open(os.path.join(out_dir, "tab_observers.tex"), "w").write("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & & & & \\multicolumn{3}{c}{Adv.\\ note} \\\\\n\\cmidrule(l){5-7}\n"
    "Observer & Self & Cross & Ind. & w/o & with & $p$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
e6w = re.findall(r"E6 world (\d+)", txt)
NUM.update({"e6.n": len(e6w) - 1, "e6.rho": one(r"Spearman\(G in-context, G weight\)=([\d.-]+)")})      # the first world only selects the learning rate

# ---- status, files
failed = re.findall(r"NOT SUPPORTED: (.*?)  \[(.*?)\]", txt)
status = ("Numbers parsed from the printed output of the executed v7 notebook (FULL mode). For exact values and vector "
          "figures, replace \\texttt{paper\\_inputs} with the folder inside \\texttt{cp\\_results.zip}.")
with open(os.path.join(out_dir, "numbers.tex"), "w") as f:
    f.write("% Written by tools/inputs_from_notebook.py from " + os.path.basename(nb_path) + "\n% Parsed from printed notebook output; the observer matrix and the "
            "heatmap were transcribed from annotated figures.\n% The notebook's own prose checks that failed in this run (the text has since been revised):\n" +
            "".join(f"%   - {c} [{w}]\n" for c, w in failed))
    for k_, v in NUM.items(): f.write(f"\\setnum{{{k_}}}{{{v}}}\n")
    f.write("\\renewcommand{\\numbersstatus}{" + status + "}\n")
# ---- figures: Figure 4 as the PNG the notebook displayed; Figure 3 redrawn from the transcribed values with the notebook's plotting code
import base64
for c in nb["cells"]:
    if c["cell_type"] == "code" and 'savefig(fig, "fig13_e8b_breakeven")' in "".join(c["source"]):
        png = [o["data"]["image/png"] for o in c.get("outputs", []) if "data" in o and "image/png" in o["data"]][0]
        open(os.path.join(out_dir, "fig13_e8b_breakeven.png"), "wb").write(base64.b64decode(png))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8985", "font.family": "DejaVu Sans"})
USEFUL = HEAT_ROWS[:4]; Hm = np.array(HEAT)
LAB = {"U": "True utility $U$", "CP": "Sealed gain", "CP_ratio": "Gain per bit", "IG_full": "Unsealed gain", "surprise": "Surprise", "relevance": "Relevance",
       "novelty": "Novelty", "judge": "LLM judge"}
div = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f1f0ec", "#2a78d6"])
fig, ax = plt.subplots(figsize=(3.5, 3.25)); vmax = np.abs(Hm).max(); ax.imshow(Hm, cmap=div, vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(HEAT_COLS))); ax.set_xticklabels([LAB[c] for c in HEAT_COLS], rotation=40, ha="right", fontsize=7)
ax.set_yticks(range(len(HEAT_ROWS))); ax.set_yticklabels([t + (" \u2714" if t in USEFUL else "") for t in HEAT_ROWS], fontsize=7)
for (r, c), v in np.ndenumerate(Hm): ax.text(c, r, f"{v:+.1f}", ha="center", va="center", fontsize=6, color="#0b0b0b" if abs(v) < 0.6 * vmax else "white")
ax.axhline(len(USEFUL) - 0.5, color="#0b0b0b", lw=0.9); ax.axvline(0.5, color="#0b0b0b", lw=0.9); ax.grid(False); ax.tick_params(length=2, pad=2)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig3_e1_heatmap.pdf"), bbox_inches="tight")
if len(sys.argv) > 4 and sys.argv[3] == "--replication":      # also write selected headline numbers under the prefix rep.
    keys = ["cfg.ntest", "e1.rho.CP", "e1.rho.best", "e1.best", "e1.p", "dose.CP", "dose.maxbase", "cert.n", "ag.best", "ag.p.bpt", "ag.p.acc", "obs.self", "obs.cross"]
    with open(sys.argv[4], "w") as f:
        f.write("% Headline numbers of an earlier full run (executed notebook " + os.path.basename(nb_path) + "), parsed from its printed output\n"
                "% by tools/inputs_from_notebook.py. Used only by the Replication paragraph.\n")
        for k_ in keys: f.write(f"\\setnum{{rep.{k_}}}{{{NUM[k_]}}}\n")
print(len(NUM), "numbers written to", out_dir)
