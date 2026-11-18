import json, os, sys
# Usage: python3 regen_bandit_figures.py path/to/compression_progress_llm_agents_v9.ipynb
nb=json.load(open(sys.argv[1]))
cells=[ "".join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
g={}
os.environ["CP_MODE"]="FULL"; os.environ["CP_RESULTS"]=os.path.abspath("out")
exec("import os\nIN_COLAB=False\n", g)
import matplotlib; matplotlib.use("Agg")
exec(cells[1], g)
exec(cells[2].replace('if not IN_COLAB and MODE == "TEST": matplotlib.use("Agg")','matplotlib.use("Agg")'), g)
import matplotlib.pyplot as plt; plt.rcParams["pdf.fonttype"]=42
exec(cells[3], g)
c5=cells[5].replace('"cp": "Compression progress, sealed audit", "cp_retro": "Compression progress, retrospective (eq. 6)"',
                    '"cp": "Sealed learning progress", "cp_retro": "Retrospective learning progress"')
assert c5!=cells[5]; exec(c5, g)
c6=cells[6].replace('"cp": "CP, sealed", "cp_retro": "CP, retrospective"','"cp": "Sealed LP", "cp_retro": "Retrospective LP"')
assert c6!=cells[6]; exec(c6, g)
exec(cells[7], g)
json.dump({"noise_share":{k:list(map(float,v)) for k,v in g["noise_share"].items()},
           "final":{k:list(map(float,v)) for k,v in g["final"].items()},
           "u":{f"{s}_T{T}":g["ushare"][(s,T)].mean(0).tolist() for (s,T) in g["ushare"]}}, open("check.json","w"), indent=1)
print("DONE")
