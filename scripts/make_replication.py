#!/usr/bin/env python3
"""Write replication.tex: headline numbers of the two earlier full runs, copied from their numbers.tex files.

    python tools/make_replication.py earlier_runs/v7_numbers.tex earlier_runs/v8_numbers.tex > replication.tex

v7_numbers.tex was parsed from the executed v7 notebook by tools/inputs_from_notebook.py (printed precision);
v8_numbers.tex is the file the v8 notebook wrote. Keys become rep.<key> (first run) and rep2.<key> (second run)."""
import re, sys
KEYS = ["cfg.ntest", "e1.rho.CP", "e1.rho.best", "e1.best", "e1.p", "dose.CP", "dose.maxbase", "cert.n", "ag.best", "ag.p.bpt"]
runs = [dict(re.findall(r"\\setnum\{([^}]+)\}\{(.*)\}\s*$", open(f).read(), re.M)) for f in sys.argv[1:3]]
assert runs[0]["e1.best"] == runs[1]["e1.best"] and runs[0]["cfg.ntest"] == runs[1]["cfg.ntest"] and runs[0]["dose.maxbase"] == runs[1]["dose.maxbase"], \
    "the Replication paragraph assumes the same best baseline, test-set size and best distractor baseline in both earlier runs"
print("% Headline numbers of the two earlier full runs (different draws of the worlds). Written by tools/make_replication.py from")
print("% " + " and ".join(sys.argv[1:3]) + ". Used only by the Replication paragraph.")
for pre, r in zip(("rep", "rep2"), runs):
    for k in KEYS: print(f"\\setnum{{{pre}.{k}}}{{{r[k]}}}")
