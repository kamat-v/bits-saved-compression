"""Build sealed-gain evaluation tasks from EvalPlus (HumanEval+ and MBPP+).

Each EvalPlus problem is a hidden Python function f with many input/output tests.
The agent's observations are calls "f(args) == output". For every problem we build
  * an exposed history (3 calls), a sealed audit (8 calls) and future calls (8),
  * candidate documents: SPEC (the natural-language task description), CODE (the
    reference implementation), EXAMPLES (6 further calls), two WRONG implementations
    (single-site mutants of CODE that change some outputs), ECHO (the exposed calls,
    reordered), two DISTRACTOR implementations of other problems, SHUFFLED (CODE's
    tokens in random order), NOISE (random base64) and BOILER (generic prose).
The function is renamed to f everywhere, and every output is computed by running the
reference implementation, so the ground truth is exact.
"""
import ast, base64, copy, gzip, json, multiprocessing as mp, re, signal

import numpy as np

N_EXP, N_AUD, N_FUT, N_EX = 3, 8, 8, 6
MAX_LINE = 160          # characters per observation line
MAX_INPUTS = 120        # inputs tried per problem (a seeded subset of EvalPlus's tests)
CALL_TIMEOUT = 1        # seconds per call
BOILER = [
    "Thank you for visiting our website. Please read our terms of service carefully before continuing. By using this site you agree to the use of cookies for analytics and personalization.",
    "Welcome to the community newsletter. This month we are excited to share several updates about upcoming events, volunteer opportunities, and ways to stay connected with your neighbors.",
    "In today's fast-paced world, it is more important than ever to stay organized. Here are a few simple tips to help you manage your time, reduce stress, and focus on what matters most.",
    "The weather this week will be mostly sunny with a few clouds in the afternoon. Temperatures will remain mild, and there is a small chance of rain over the weekend.",
]


# ---------------------------------------------------------------- loading and renaming
def load_problems(path):
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def rename(text, old, new="f"):
    return re.sub(r"\b%s\b" % re.escape(old), new, text)


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
        return node
    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _strip


def full_code(p):
    """Reference implementation with the entry point renamed to f (HumanEval: prompt + body; MBPP: solution)."""
    src = p["prompt"] + p["canonical_solution"] if p["task_id"].startswith("HumanEval") else p["canonical_solution"]
    return rename(src, p["entry_point"])


def code_doc(src):
    """CODE candidate: the implementation without docstrings, in a normalized layout (ast.unparse)."""
    tree = _StripDocstrings().visit(ast.parse(src))
    return ast.unparse(tree)


def spec_doc(p):
    """SPEC candidate: the natural-language task description, renamed."""
    if p["task_id"].startswith("HumanEval"):
        return rename(p["prompt"], p["entry_point"]).strip()
    return rename(p["prompt"], p["entry_point"]).strip().strip('"').strip()


# ---------------------------------------------------------------- execution (in a child process)
class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _run_calls(src, inputs, conn):
    """Execute src, then call f on each input; send back repr(output) or None per input."""
    out = []
    try:
        signal.signal(signal.SIGALRM, _alarm)
        g = {"__name__": "__evalplus__"}
        signal.alarm(5)
        exec(compile(src, "<cand>", "exec"), g)
        signal.alarm(0)
        f = g["f"]
        for args in inputs:
            try:
                signal.alarm(CALL_TIMEOUT)
                r = f(*copy.deepcopy(args))
                signal.alarm(0)
                s = repr(r)
                out.append(s if len(s) < 400 else None)
            except BaseException:
                signal.alarm(0)
                out.append(None)
    except BaseException:
        out = [None] * len(inputs)
    conn.send(out)
    conn.close()


def run_calls(src, inputs, timeout=60):
    parent, child = mp.Pipe(duplex=False)
    proc = mp.Process(target=_run_calls, args=(src, inputs, child), daemon=True)
    proc.start()
    child.close()
    res = None
    if parent.poll(timeout):
        try:
            res = parent.recv()
        except EOFError:
            res = None
    proc.join(1)
    if proc.is_alive():
        proc.kill()
    return res if res is not None else [None] * len(inputs)


# ---------------------------------------------------------------- mutants
_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv,
         ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
         ast.And: ast.Or, ast.Or: ast.And}


def _sites(tree):
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in _SWAP:
            sites.append(("binop", node))
        elif isinstance(node, ast.Compare) and node.ops and type(node.ops[0]) in _SWAP:
            sites.append(("cmp", node))
        elif isinstance(node, ast.BoolOp) and type(node.op) in _SWAP:
            sites.append(("bool", node))
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            sites.append(("int", node))
        elif isinstance(node, ast.Constant) and type(node.value) is bool:
            sites.append(("boolc", node))
    return sites


def mutants(code, rng, limit=40):
    """Single-site mutants of the normalized CODE text, in a seeded random order."""
    n = len(_sites(ast.parse(code)))
    order = rng.permutation(n)[:limit]
    out = []
    for k in order:
        tree = ast.parse(code)
        kind, node = _sites(tree)[k]
        if kind == "binop":
            node.op = _SWAP[type(node.op)]()
        elif kind == "cmp":
            node.ops[0] = _SWAP[type(node.ops[0])]()
        elif kind == "bool":
            node.op = _SWAP[type(node.op)]()
        elif kind == "int":
            node.value = node.value + 1
        elif kind == "boolc":
            node.value = not node.value
        m = ast.unparse(tree)
        if m != code:
            out.append(m)
    return out


# ---------------------------------------------------------------- task construction
def args_text(args):
    return ", ".join(repr(a) for a in args)


def obs_pieces(args, out):
    """One observation line as (text, tag) pieces; tag 1 marks the output, used for exact-output accuracy."""
    return [(f"f({args_text(args)}) == ", 0), (out, 1), ("\n", 0)]


def line_text(pieces):
    return "".join(t for t, _ in pieces)


def build_task(p, others, seed):
    rng = np.random.default_rng(seed)
    src = full_code(p)
    try:
        code = code_doc(src)
    except SyntaxError:
        return None
    spec = spec_doc(p)
    inputs = list(p["base_input"]) + list(p["plus_input"])
    idx = rng.permutation(len(inputs))[:MAX_INPUTS]
    inputs = [inputs[i] for i in idx]
    outs = run_calls(code, inputs)
    outs2 = run_calls(code, inputs)                     # determinism check
    tests, seen = [], set()
    for a, o, o2 in zip(inputs, outs, outs2):
        if o is None or o != o2:
            continue
        at = args_text(a)
        line = f"f({at}) == {o}"
        if len(line) > MAX_LINE or at in seen or f"f({at})".replace(" ", "") in spec.replace(" ", "").replace("\n", ""):
            continue                                   # too long, duplicate, or quoted in the task description (could be copied)
        seen.add(at)
        tests.append((a, o))
    need = N_EXP + N_AUD + N_FUT + N_EX
    if len(tests) < need:
        return None
    exp, aud = tests[:N_EXP], tests[N_EXP:N_EXP + N_AUD]
    fut, ex = tests[N_EXP + N_AUD:N_EXP + N_AUD + N_FUT], tests[N_EXP + N_AUD + N_FUT:need]
    held = aud + fut
    wrong = []
    for m in mutants(code, rng):
        mo = run_calls(m, [a for a, _ in exp + held], timeout=30)
        if any(o is None for o in mo):
            continue
        diff_held = sum(o != t for o, (_, t) in zip(mo[N_EXP:], held))
        diff_aud = sum(o != t for o, (_, t) in zip(mo[N_EXP:N_EXP + N_AUD], aud))
        if diff_held >= 2 and diff_aud >= 1:
            wrong.append(m)
        if len(wrong) == 2:
            break
    if len(wrong) < 2:
        return None
    echo = [line_text(obs_pieces(a, o)) for a, o in exp]
    order = rng.permutation(len(echo))
    if list(order) == list(range(len(echo))):
        order = order[::-1]
    words = code.split()
    rng.shuffle(words)
    docs = {
        "SPEC": spec,
        "CODE": code,
        "EXAMPLES": "More calls:\n" + "".join(line_text(obs_pieces(a, o)) for a, o in ex).strip(),
        "WRONG_1": wrong[0],
        "WRONG_2": wrong[1],
        "ECHO": "Log excerpt:\n" + "".join(echo[i] for i in order).strip(),
        "DISTRACTOR_1": others[0],
        "DISTRACTOR_2": others[1],
        "SHUFFLED": " ".join(words),
        "NOISE": " ".join(base64.b64encode(rng.bytes(len(code))).decode()[i:i + 12] for i in range(0, len(code), 12)),
        "BOILER": BOILER[int(rng.integers(len(BOILER)))],
    }
    return dict(task_id=p["task_id"], exp=[obs_pieces(a, o) for a, o in exp], aud=[obs_pieces(a, o) for a, o in aud],
                fut=[obs_pieces(a, o) for a, o in fut], docs=docs)


USEFUL_TYPES = ["SPEC", "CODE", "EXAMPLES"]
CONTROL_TYPES = ["WRONG", "ECHO", "DISTRACTOR", "SHUFFLED", "NOISE", "BOILER"]


def dtype_of(key):
    return key.split("_")[0]


def build_all(paths, seed=0, limit=None, log=print):
    probs = []
    for path in paths:
        probs += load_problems(path)
    if limit:
        probs = probs[:limit]
    codes = {}
    for p in probs:
        try:
            codes[p["task_id"]] = code_doc(full_code(p))
        except SyntaxError:
            pass
    tasks = []
    ids = sorted(codes)
    for i, p in enumerate(probs):
        if p["task_id"] not in codes:
            continue
        rng = np.random.default_rng(seed + 1000 + i)
        pool = [t for t in ids if t != p["task_id"] and t.split("/")[0] == p["task_id"].split("/")[0]]
        others = [codes[pool[j]] for j in rng.choice(len(pool), 2, replace=False)]
        t = build_task(p, others, seed + i)
        if t is not None:
            tasks.append(t)
        if log and (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(probs)} problems examined, {len(tasks)} tasks built")
    return tasks


if __name__ == "__main__":
    # Rebuild evalplus_tasks.json.gz from the EvalPlus release files:
    #   git clone https://github.com/evalplus/humanevalplus_release
    #   git clone https://github.com/evalplus/mbppplus_release
    #   python3 evalplus_tasks.py humanevalplus_release/HumanEvalPlus.jsonl.gz mbppplus_release/MbppPlus.jsonl.gz
    import sys
    tasks = []
    for path in sys.argv[1:]:
        tasks += build_all([path], log=None)
    with gzip.open("evalplus_tasks.json.gz", "wt") as fh:
        json.dump(tasks, fh)
    print(len(tasks), "tasks written to evalplus_tasks.json.gz")
