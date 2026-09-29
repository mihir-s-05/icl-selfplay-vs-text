"""In-context learning task generators.

Two suites:

* ``raw``: a verbatim port of the paper's harness
  (``figures/fig5_icl_sum_behavior/icl_harness/run_icl*.py``). Same RNG seeds,
  same draw order, same skip rules, so every prompt is byte-identical to the
  paper's. ``tests/test_prompt_parity.py`` checks this against the original
  scripts. Inputs are uniform bytes 1..255 with sentinel byte 0, a format that
  is off-distribution for a natural-text model.

* ``printable``: the same task families rendered in printable ASCII
  (lowercase letters, decimal digits, ``\\n`` separator, ``=`` before the
  answer), plus a substitution-cipher task. This is the fair format for a model
  pretrained on text.

Each generator yields :class:`Cell` objects. A cell holds a batch of trials:
the token sequences, and per trial the indices whose logits are read
(``score_at``: logits at index ``i`` predict token ``i + 1``) and the target at
each. Single-answer tasks score one position at the end of the prompt;
teacher-forced tasks (reverse string, multi-digit sums, cipher) score several.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

O = ord("O")
PUSH, POP = 250, 251


@dataclass
class Cell:
    section: str                 # results group (mirrors the paper's files)
    tag: str                     # key within the section (the paper's key)
    seqs: list                   # [N] token lists (ragged allowed)
    score_at: list               # [N] lists of logit indices to score
    targets: list                # [N] lists of target bytes (same lengths)
    chance: float = 1 / 256
    extra: dict = field(default_factory=dict)

    @property
    def cid(self) -> str:
        return f"{self.section}/{self.tag}"

    @property
    def max_len(self) -> int:
        return max(len(s) for s in self.seqs)


def _last(seqs, targets, section, tag, **kw) -> Cell:
    """Single-answer cell: the answer is the byte after the prompt."""
    return Cell(section, tag, seqs, [[len(s) - 1] for s in seqs],
                [[int(t)] for t in targets], **kw)


def _tail(seqs, answers, section, tag, **kw) -> Cell:
    """Teacher-forced cell: each seq ends with its answer; score every answer byte."""
    score_at, targets = [], []
    for s, a in zip(seqs, answers):
        a = [int(v) for v in a]
        T = len(s)
        score_at.append([T - len(a) + j - 1 for j in range(len(a))])
        targets.append(a)
    return Cell(section, tag, seqs, score_at, targets, **kw)


# ============================================================ raw (paper) suite

SWEEP_FUNCS: dict[str, Callable] = {
    "max": lambda x: int(np.max(x)),
    "min": lambda x: int(np.min(x)),
    "sum": lambda x: int(np.sum(x) % 256),
    "mean": lambda x: int(np.mean(x)),
    "first": lambda x: int(x[0]),
    "last": lambda x: int(x[-1]),
}
SWEEP_MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
SWEEP_KS = [2, 4, 8, 16]


def raw_sweep() -> Iterator[Cell]:
    """run_icl.py: max/min/sum/mean/first/last over (m, k); rng 0, 64 trials."""
    rng = np.random.default_rng(0)
    for fname, f in SWEEP_FUNCS.items():
        for m_ in SWEEP_MS:
            for k in SWEEP_KS:
                if 1 + m_ * (k + 2) + k + 1 > 4096:
                    continue
                seqs, targets = [], []
                for _ in range(64):
                    toks = [O]
                    for _ in range(m_):
                        x = rng.integers(1, 256, k)
                        toks += [0] + x.tolist() + [f(x)]
                    xq = rng.integers(1, 256, k)
                    toks += [0] + xq.tolist()
                    seqs.append(toks)
                    targets.append(f(xq))
                yield _last(seqs, targets, "sweep", f"{fname}|{m_}|{k}")


def raw_v2() -> Iterator[Cell]:
    """run_icl_v2.py: index, shift, sum/diff/xor; rng 3, 128 trials."""
    rng = np.random.default_rng(3)
    T = 128
    for k in (2, 4, 8, 16):
        for m_ in (4, 16, 64, 256):
            if 1 + m_ * (k + 3) + k + 2 > 4096:
                continue
            seqs, targets, sup = [], [], []
            for _ in range(T):
                toks = [O]
                for _ in range(m_):
                    x = rng.integers(1, 256, k)
                    i = int(rng.integers(1, k + 1))
                    toks += [0] + x.tolist() + [i, int(x[i - 1])]
                xq = rng.integers(1, 256, k)
                iq = int(rng.integers(1, k + 1))
                toks += [0] + xq.tolist() + [iq]
                seqs.append(toks); targets.append(int(xq[iq - 1])); sup.append(sorted(set(xq.tolist())))
            yield _last(seqs, targets, "v2", f"index|m={m_}|k={k}", extra={"supports": sup})
    for m_ in (2, 4, 8, 16, 32, 128, 512):
        seqs, targets, sup = [], [], []
        for _ in range(T):
            c = int(rng.integers(1, 256))
            toks = [O]
            for _ in range(m_):
                x = int(rng.integers(1, 256))
                toks += [0, x, (x + c) % 256]
            xq = int(rng.integers(1, 256))
            toks += [0, xq]
            seqs.append(toks); targets.append((xq + c) % 256); sup.append([xq])
        yield _last(seqs, targets, "v2", f"shift|m={m_}|k=1", extra={"supports": sup})
    ops = {"sum": lambda a, b: (a + b) % 256, "diff": lambda a, b: (a - b) % 256,
           "xor": lambda a, b: a ^ b}
    for op, f in ops.items():
        for m_ in (8, 32, 128, 512):
            seqs, targets, sup = [], [], []
            for _ in range(T):
                toks = [O]
                for _ in range(m_):
                    a, b = (int(v) for v in rng.integers(1, 256, 2))
                    toks += [0, a, b, f(a, b)]
                a, b = (int(v) for v in rng.integers(1, 256, 2))
                toks += [0, a, b]
                seqs.append(toks); targets.append(f(a, b)); sup.append(sorted({a, b}))
            yield _last(seqs, targets, "v2", f"{op}|m={m_}|k=2", extra={"supports": sup})


def raw_v3() -> Iterator[Cell]:
    """run_icl_v3.py: assoc, succ/compl/double, absdiff, closest, prev; rng 4."""
    rng = np.random.default_rng(4)
    T, MS = 128, [8, 32, 128, 512]
    for V in (4, 16, 64):
        for m_ in MS:
            if m_ < V or 1 + m_ * 3 + 2 > 4096:
                continue
            seqs, targets = [], []
            for _ in range(T):
                keys = rng.choice(np.arange(1, 256), V, replace=False)
                vals = rng.integers(1, 256, V)
                toks = [O]
                shown = set()
                for j in range(m_):
                    i = int(rng.integers(0, V)) if j >= V else j
                    toks += [0, int(keys[i]), int(vals[i])]
                    shown.add(i)
                iq = int(rng.choice(sorted(shown)))
                toks += [0, int(keys[iq])]
                seqs.append(toks); targets.append(int(vals[iq]))
            yield _last(seqs, targets, "v3", f"assoc|m={m_}|V={V}")
    maps = {"succ": lambda x: (x + 1) % 256, "compl": lambda x: 255 - x,
            "double": lambda x: (2 * x) % 256}
    for name, f in maps.items():
        for m_ in MS:
            seqs, targets = [], []
            for _ in range(T):
                toks = [O]
                for _ in range(m_):
                    x = int(rng.integers(1, 256))
                    toks += [0, x, f(x)]
                xq = int(rng.integers(1, 256))
                toks += [0, xq]
                seqs.append(toks); targets.append(f(xq))
            yield _last(seqs, targets, "v3", f"{name}|m={m_}|k=1")
    for m_ in MS:
        seqs, targets = [], []
        for _ in range(T):
            toks = [O]
            for _ in range(m_):
                a, b = (int(v) for v in rng.integers(1, 256, 2))
                toks += [0, a, b, abs(a - b)]
            a, b = (int(v) for v in rng.integers(1, 256, 2))
            toks += [0, a, b]
            seqs.append(toks); targets.append(abs(a - b))
        yield _last(seqs, targets, "v3", f"absdiff|m={m_}|k=2")
    for m_ in MS:
        k = 4
        if 1 + m_ * (k + 3) + k + 2 > 4096:
            continue
        seqs, targets = [], []

        def ex():
            x = rng.integers(1, 256, k)
            q = int(rng.integers(1, 256))
            y = int(x[np.argmin(np.abs(x.astype(int) - q))])
            return x, q, y
        for _ in range(T):
            toks = [O]
            for _ in range(m_):
                x, q, y = ex()
                toks += [0] + x.tolist() + [q, y]
            x, q, y = ex()
            toks += [0] + x.tolist() + [q]
            seqs.append(toks); targets.append(y)
        yield _last(seqs, targets, "v3", f"closest|m={m_}|k=4")
    for m_ in MS:
        seqs, targets = [], []
        for _ in range(T):
            toks = [O]
            prev = int(rng.integers(1, 256))
            toks += [0, prev, int(rng.integers(1, 256))]
            for _ in range(m_ - 1):
                x = int(rng.integers(1, 256))
                toks += [0, x, prev]
                prev = x
            xq = int(rng.integers(1, 256))
            toks += [0, xq]
            seqs.append(toks); targets.append(prev)
        yield _last(seqs, targets, "v3", f"prev|m={m_}|k=1")


def _stack_trace(rng, L, with_depth=True):
    """Random valid stack program of L ops ending in POP (paper's ``trace``)."""
    while True:
        toks, stack = [], []
        for _ in range(L - 1):
            if stack and rng.random() < 0.5:
                v = stack.pop()
                toks += [POP, v]
            else:
                v = int(rng.integers(1, 250))
                stack.append(v)
                toks += [PUSH, v]
        if stack:
            ans, depth = stack[-1], len(stack)
            toks += [POP]
            return (toks, ans, depth) if with_depth else (toks, ans)


def raw_v4() -> Iterator[Cell]:
    """run_icl_v4.py: reverse string (teacher-forced) and stack; rng 5."""
    rng = np.random.default_rng(5)
    T = 128
    for k in (2, 4, 8, 16):
        for m_ in (1, 2, 4, 8, 32, 128):
            unit = 2 * k + 1
            if 1 + (m_ + 1) * unit > 4096:
                continue
            seqs, ans = [], []
            for _ in range(T):
                toks = [O]
                for _ in range(m_):
                    x = rng.integers(1, 256, k)
                    toks += [0] + x.tolist() + x[::-1].tolist()
                xq = rng.integers(1, 256, k)
                toks += [0] + xq.tolist() + xq[::-1].tolist()
                seqs.append(toks); ans.append(xq[::-1].tolist())
            yield _tail(seqs, ans, "v4", f"palin|m={m_}|k={k}")
    for L in (4, 8, 16):
        for m_ in (1, 2, 4, 8, 32, 128):
            if 1 + (m_ + 1) * (1 + 2 * L) > 4096:
                continue
            seqs, targets, dep = [], [], []
            for _ in range(T):
                toks = [O]
                for _ in range(m_):
                    tr, a, _ = _stack_trace(rng, L)
                    toks += [0] + tr + [a]
                tr, a, d = _stack_trace(rng, L)
                toks += [0] + tr
                seqs.append(toks); targets.append(a); dep.append(d)
            yield _last(seqs, targets, "v4", f"stack|m={m_}|L={L}", extra={"depth": dep})


def raw_extras() -> Iterator[Cell]:
    """run_icl_extras.py: m=0 cells, 4096-trial low-m sum, assoc by extra repeats."""
    rng = np.random.default_rng(0)
    N0 = 1024
    k = 8
    seqs, ans = [], []
    for _ in range(N0):
        xq = rng.integers(1, 256, k)
        seqs.append([O, 0] + xq.tolist() + xq[::-1].tolist())
        ans.append(xq[::-1].tolist())
    yield _tail(seqs, ans, "m0", f"palin|m=0|k={k}")
    L = 4
    seqs, targets = [], []
    for _ in range(N0):
        tr, a = _stack_trace(rng, L, with_depth=False)
        seqs.append([O, 0] + tr); targets.append(a)
    yield _last(seqs, targets, "m0", f"stack|m=0|L={L}")
    for fname, f in (("max", lambda x: int(x.max())), ("min", lambda x: int(x.min()))):
        seqs, targets = [], []
        for _ in range(N0):
            xq = rng.integers(1, 256, 2)
            seqs.append([O, 0] + xq.tolist()); targets.append(f(xq))
        yield _last(seqs, targets, "m0", f"{fname}|0|2")

    rng = np.random.default_rng(0)
    for m_ in (0, 1, 2, 3, 4):
        seqs, targets = [], []
        for _ in range(4096):
            toks = [O]
            for _ in range(m_):
                x = rng.integers(1, 256, 2)
                toks += [0] + x.tolist() + [int(x.sum()) % 256]
            xq = rng.integers(1, 256, 2)
            toks += [0] + xq.tolist()
            seqs.append(toks); targets.append(int(xq.sum()) % 256)
        yield _last(seqs, targets, "sum_lowm", f"sum|{m_}|2")

    rng = np.random.default_rng(4)
    V = 16
    for extra in (0, 1, 2, 4, 8, 16):
        seqs, targets = [], []
        for _ in range(1024):
            keys = rng.choice(np.arange(1, 256), V, replace=False)
            vals = rng.integers(1, 256, V)
            toks = [O]
            for j in range(V):
                toks += [0, int(keys[j]), int(vals[j])]
            for _ in range(extra):
                i = int(rng.integers(0, V))
                toks += [0, int(keys[i]), int(vals[i])]
            iq = int(rng.integers(0, V))
            toks += [0, int(keys[iq])]
            seqs.append(toks); targets.append(int(vals[iq]))
        yield _last(seqs, targets, "assoc_dict", f"assoc|extra={extra}|V={V}")


def raw_control() -> Iterator[Cell]:
    """run_icl_control.py: decoupled control; rng 1, 256 trials.

    Each cell holds 2N sequences: N normal prompts then N decoupled prompts
    (same demos, unrelated shown query). Targets are f(real query) for both.
    """
    rng = np.random.default_rng(1)
    T = 256
    funcs = {"max": lambda x: int(np.max(x)), "min": lambda x: int(np.min(x)),
             "mean": lambda x: int(np.mean(x)), "sum": lambda x: int(np.sum(x) % 256)}
    cells = [(f, m, 2) for f in funcs for m in (32, 128, 256, 512)] + \
            [("max", 128, 4), ("sum", 128, 4)]
    for fname, m_, k in cells:
        if 1 + m_ * (k + 2) + k + 1 > 4096:
            continue
        f = funcs[fname]
        normal, decoup, targets, queries = [], [], [], []
        for _ in range(T):
            demo = []
            for _ in range(m_):
                x = rng.integers(1, 256, k)
                demo += [0] + x.tolist() + [f(x)]
            xq = rng.integers(1, 256, k)
            xs = rng.integers(1, 256, k)
            normal.append([O] + demo + [0] + xq.tolist())
            decoup.append([O] + demo + [0] + xs.tolist())
            targets.append(f(xq)); queries.append(xq.tolist())
        yield _last(normal + decoup, targets + targets, "control", f"{fname}|{m_}|{k}",
                    extra={"n_normal": T, "queries": queries})


RAW_SECTIONS = {"sweep": raw_sweep, "v2": raw_v2, "v3": raw_v3, "v4": raw_v4,
                "extras": raw_extras, "control": raw_control}
# The sections Figure 4 reads (sweep, v3, v4 and the extras).
FIG4_SECTIONS = ("sweep", "v3", "v4", "extras")


# ======================================================== printable-text suite

A, Z = ord("a"), ord("z")
NL, EQ = ord("\n"), ord("=")
LETTERS = np.arange(A, Z + 1)
P_MS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
P_TRIALS = 128


def _letters(rng, n):
    return rng.integers(A, Z + 1, n)


def _enc(s: str) -> list:
    return list(s.encode("ascii"))


def printable() -> Iterator[Cell]:
    """Fig. 4 task families in printable ASCII, plus a substitution cipher.

    Prompt = ``O`` then m demos ``\\n<input>=<answer>`` then the query
    ``\\n<input>=``. Multi-byte answers are teacher-forced and a trial counts
    as correct only if every answer byte is (``acc``); ``acc_mean`` is the
    per-byte rate. Chance is 1/26 per letter and 1/10 per digit.
    """
    rng = np.random.default_rng(1000)
    lc = 1 / 26

    def fits(m_, unit):
        return 1 + (m_ + 1) * unit <= 4096

    # --- first / last / max / min over k=2 letters (alphabetical order)
    fns = {"first": lambda x: int(x[0]), "last": lambda x: int(x[-1]),
           "max": lambda x: int(x.max()), "min": lambda x: int(x.min())}
    for fname, f in fns.items():
        for m_ in P_MS:
            seqs, targets = [], []
            for _ in range(P_TRIALS):
                toks = [O]
                for _ in range(m_):
                    x = _letters(rng, 2)
                    toks += [NL] + x.tolist() + [EQ, f(x)]
                xq = _letters(rng, 2)
                toks += [NL] + xq.tolist() + [EQ]
                seqs.append(toks); targets.append(f(xq))
            yield _last(seqs, targets, "printable", f"{fname}|m={m_}|k=2", chance=lc)

    # --- sum of two decimals 0..99, answer teacher-forced digit by digit
    for m_ in P_MS:
        if not fits(m_, 10):
            continue
        seqs, ans = [], []
        for _ in range(P_TRIALS):
            toks = [O]
            for _ in range(m_):
                a, b = (int(v) for v in rng.integers(0, 100, 2))
                toks += _enc(f"\n{a}+{b}={a + b}")
            a, b = (int(v) for v in rng.integers(0, 100, 2))
            toks += _enc(f"\n{a}+{b}={a + b}")
            seqs.append(toks); ans.append(_enc(str(a + b)))
        yield _tail(seqs, ans, "printable", f"sum|m={m_}|k=2", chance=0.1)

    # --- reverse k=8 letters, teacher-forced
    k = 8
    for m_ in P_MS:
        if not fits(m_, 2 * k + 2):
            continue
        seqs, ans = [], []
        for _ in range(P_TRIALS):
            toks = [O]
            for _ in range(m_):
                x = _letters(rng, k)
                toks += [NL] + x.tolist() + [EQ] + x[::-1].tolist()
            xq = _letters(rng, k)
            toks += [NL] + xq.tolist() + [EQ] + xq[::-1].tolist()
            seqs.append(toks); ans.append(xq[::-1].tolist())
        yield _tail(seqs, ans, "printable", f"reverse|m={m_}|k={k}", chance=lc)

    # --- stack, L=4 ops: "+x" pushes letter x, "-x" pops x; query ends "-"
    L = 4
    for m_ in P_MS:
        if not fits(m_, 2 * L + 2):
            continue
        seqs, targets = [], []
        for _ in range(P_TRIALS):
            toks = [O]
            for j in range(m_ + 1):
                while True:
                    ops, stack = [], []
                    for _ in range(L - 1):
                        if stack and rng.random() < 0.5:
                            ops += [ord("-"), stack.pop()]
                        else:
                            v = int(rng.integers(A, Z + 1))
                            stack.append(v)
                            ops += [ord("+"), v]
                    if stack:
                        break
                toks += [NL] + ops + [ord("-")]
                if j < m_:
                    toks += [stack[-1]]
            seqs.append(toks); targets.append(stack[-1])
        yield _last(seqs, targets, "printable", f"stack|m={m_}|L={L}", chance=lc)

    # --- associative recall, V=16 distinct letter keys: one full printing of
    #     the dictionary, then `extra` random repeats, then a query
    V = 16
    for extra in (0, 1, 2, 4, 8, 16, 48, 112):
        seqs, targets = [], []
        for _ in range(P_TRIALS):
            keys = rng.choice(LETTERS, V, replace=False)
            vals = _letters(rng, V)
            order = list(range(V)) + [int(i) for i in rng.integers(0, V, extra)]
            toks = [O]
            for i in order:
                toks += [NL, int(keys[i]), EQ, int(vals[i])]
            iq = int(rng.integers(0, V))
            toks += [NL, int(keys[iq]), EQ]
            seqs.append(toks); targets.append(int(vals[iq]))
        yield _last(seqs, targets, "printable", f"assoc|extra={extra}|V={V}", chance=lc)

    # --- substitution cipher: a random letter permutation per trial; demos map
    #     4-letter strings through it; the query is teacher-forced. Letters of the
    #     query may not have appeared in the demos, so acc is capped for small m.
    k = 4
    for m_ in P_MS:
        if not fits(m_, 2 * k + 2):
            continue
        seqs, ans, seen_frac = [], [], []
        for _ in range(P_TRIALS):
            perm = rng.permutation(26)
            enc = lambda x: [A + int(perm[c - A]) for c in x]
            toks, seen = [O], set()
            for _ in range(m_):
                x = _letters(rng, k).tolist()
                toks += [NL] + x + [EQ] + enc(x)
                seen.update(x)
            xq = _letters(rng, k).tolist()
            toks += [NL] + xq + [EQ] + enc(xq)
            seqs.append(toks); ans.append(enc(xq))
            seen_frac.append(float(np.mean([c in seen for c in xq])))
        yield _tail(seqs, ans, "printable", f"cipher|m={m_}|k={k}", chance=lc,
                    extra={"query_letters_seen": float(np.mean(seen_frac))})


# ============================================================== word-level suite

# Common English words in 10 semantic categories (20 each, no word in two
# categories). Self-play never saw language, so it can only use word identity;
# a text model can also use meaning.
WORD_CATEGORIES = {
    "animal": "cat dog horse cow sheep goat pig lion tiger bear wolf fox mouse rabbit "
              "deer zebra monkey camel whale snake",
    "fruit": "apple banana grape lemon cherry peach pear plum mango melon berry kiwi lime "
             "apricot fig coconut papaya guava date orange",
    "color": "red blue green yellow purple pink brown black white gray violet indigo "
             "maroon teal navy beige cyan magenta crimson gold",
    "body": "head hand foot arm leg eye ear nose mouth knee elbow finger toe neck chest "
            "back shoulder hip wrist ankle",
    "vehicle": "car bus truck train plane boat bike ship taxi van tram jeep yacht scooter "
               "tractor rocket subway ferry wagon canoe",
    "clothing": "shirt pants dress skirt coat jacket hat sock shoe scarf glove boot belt "
                "vest sweater hoodie jeans shorts tie blouse",
    "kitchen": "spoon fork knife plate bowl cup pan pot kettle oven stove fridge mug tray "
               "ladle whisk grater jar lid sink",
    "country": "france spain italy japan china india brazil canada mexico egypt kenya peru "
               "chile norway sweden greece turkey russia poland germany",
    "job": "doctor nurse teacher farmer pilot chef lawyer baker painter singer dancer "
           "driver miner sailor soldier judge artist writer actor banker",
    "weather": "rain snow wind storm fog cloud hail sleet frost thunder lightning drizzle "
               "mist breeze gale sunshine heat cold humid dew",
}
WORD_CATEGORIES = {k: v.split() for k, v in WORD_CATEGORIES.items()}
W_MS = (0, 1, 2, 4, 8, 16, 32, 64, 128)


def words() -> Iterator[Cell]:
    """Word-level ICL: arbitrary-label word classification and word-order rules.

    * ``wordcls``  C semantic categories get C random letter labels per trial;
      demos ``\\n<word>=<label>``; the query word was NOT shown, so the label
      must come from the word's meaning (the text model's home ground).
    * ``wordseen`` same, but the query word appeared in the demos: pure lookup.
    * ``wordrev``  a 3-word phrase is written in reverse word order
      (``\\nred car tiger=tiger car red``); teacher-forced, whole answer exact.
    Chance: 1/C if the model restricts itself to the shown labels.
    """
    rng = np.random.default_rng(2000)
    cats = list(WORD_CATEGORIES)
    letters = [chr(c) for c in range(A, Z + 1)]

    def fits(toks):
        return len(toks) <= 4096

    for seen in (False, True):
        name = "wordseen" if seen else "wordcls"
        for C in (2, 4):
            for m_ in W_MS:
                if seen and m_ == 0:
                    continue
                seqs, targets = [], []
                for _ in range(P_TRIALS):
                    cs = rng.choice(cats, C, replace=False)
                    labels = rng.choice(letters, C, replace=False)
                    qc = int(rng.integers(0, C))
                    qword = str(rng.choice(WORD_CATEGORIES[cs[qc]]))
                    demos = []
                    for j in range(m_):
                        c = int(rng.integers(0, C))
                        pool = [w for w in WORD_CATEGORIES[cs[c]] if seen or w != qword]
                        demos.append((str(rng.choice(pool)), c))
                    if seen:                          # make sure the query word is shown
                        demos[int(rng.integers(0, m_))] = (qword, qc)
                    text = "".join(f"\n{w}={labels[c]}" for w, c in demos) + f"\n{qword}="
                    seqs.append([O] + _enc(text)); targets.append(ord(labels[qc]))
                if fits(max(seqs, key=len)):
                    yield _last(seqs, targets, "words", f"{name}|m={m_}|C={C}", chance=1 / C)

    allw = [w for ws in WORD_CATEGORIES.values() for w in ws]
    for m_ in W_MS:
        seqs, ans = [], []
        for _ in range(P_TRIALS):
            text = ""
            for _ in range(m_ + 1):
                ph = [str(w) for w in rng.choice(allw, 3, replace=False)]
                text += f"\n{' '.join(ph)}={' '.join(ph[::-1])}"
            seqs.append([O] + _enc(text)); ans.append(_enc(" ".join(ph[::-1])))
        if fits(max(seqs, key=len)):
            yield _tail(seqs, ans, "words", f"wordrev|m={m_}|n=3", chance=0.0)


SUITES = {**{f"raw.{k}": v for k, v in RAW_SECTIONS.items()}, "printable": printable,
          "words": words}


def resolve(spec: str) -> list[str]:
    """``raw`` / ``fig4`` / ``printable`` / ``raw.v3`` ... -> generator names."""
    out = []
    for s in spec.split(","):
        s = s.strip()
        if s == "raw":
            out += [f"raw.{k}" for k in RAW_SECTIONS]
        elif s == "fig4":
            out += [f"raw.{k}" for k in FIG4_SECTIONS]
        elif s == "all":
            out += list(SUITES)
        elif s in SUITES:
            out.append(s)
        else:
            raise SystemExit(f"unknown suite {s!r}; choose from raw, fig4, all, {', '.join(SUITES)}")
    return list(dict.fromkeys(out))
