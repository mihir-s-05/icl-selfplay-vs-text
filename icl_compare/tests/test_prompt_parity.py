"""Prompt parity: ``icl.tasks`` raw suite vs the paper's original harness scripts.

Runs *copies* of ``run_icl*.py`` (the originals write results next to
themselves) against a stub ``ProgramLanguageModel`` that records a hash of
every input row it is given and returns uniform logits, then checks that our
generators produce exactly the same multiset of prompts for each script.

    python tests/test_prompt_parity.py            # all scripts (a few minutes, CPU)
    python tests/test_prompt_parity.py v4 extras  # a subset
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from icl import tasks as T  # noqa: E402

HARNESS = HERE.parents[1] / "self_play_pretraining/figures/fig5_icl_sum_behavior/icl_harness"
SCRIPTS = {"sweep": "run_icl.py", "v2": "run_icl_v2.py", "v3": "run_icl_v3.py",
           "v4": "run_icl_v4.py", "extras": "run_icl_extras.py",
           "control": "run_icl_control.py"}

STUB = '''
import hashlib, torch
import numpy as np
SEEN = []
class ProgramLanguageModel(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
    def load_state_dict(self, *a, **k):
        return [], []
    def forward(self, input_ids=None, **kw):
        a = input_ids.cpu().numpy().astype(np.int64)
        for row in a:
            SEEN.append(hashlib.sha1(row.tobytes()).hexdigest())
        B, L = input_ids.shape
        return torch.zeros(1, 1, 256).expand(B, L, 256), None
'''


def row_hash(seq) -> str:
    return hashlib.sha1(np.asarray(seq, dtype=np.int64).tobytes()).hexdigest()


def run_original(name: str, work: Path) -> Counter:
    stub_repo = work / "stub"
    (stub_repo / "src/framework").mkdir(parents=True, exist_ok=True)
    for d in ("src", "src/framework"):
        (stub_repo / d / "__init__.py").write_text("")
    (stub_repo / "src/framework/model.py").write_text(STUB)
    ens = work / "ensemble"
    (ens / "seed-1").mkdir(parents=True, exist_ok=True)
    (ens / "config.json").write_text(json.dumps({"learner": {}}))
    import torch
    torch.save({}, ens / "seed-1" / "learner_2816.pth")
    script = work / SCRIPTS[name]
    shutil.copy(HARNESS / SCRIPTS[name], script)
    os.environ.update(SOLOMONOFF_REPO=str(stub_repo), ICL_ENSEMBLE=str(ens),
                      ICL_OUT=str(work), CUDA_VISIBLE_DEVICES="")
    for mod in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[mod]
    sys.path.insert(0, str(stub_repo))
    try:
        runpy.run_path(str(script), run_name="__main__")
        from src.framework import model as stub
        return Counter(stub.SEEN)
    finally:
        sys.path.remove(str(stub_repo))


def ours(name: str) -> Counter:
    return Counter(row_hash(s) for c in T.RAW_SECTIONS[name]() for s in c.seqs)


def main(names):
    ok = True
    for name in names:
        with tempfile.TemporaryDirectory() as tmp:
            theirs = run_original(name, Path(tmp))
        mine = ours(name)
        # Exact multiset equality: every prompt, with its multiplicity.
        same = theirs == mine
        print(f"{name:8s} original {sum(theirs.values()):6d} rows ({len(theirs)} distinct), "
              f"ours {sum(mine.values()):6d} rows ({len(mine)} distinct) -> "
              f"{'MATCH' if same else 'DIFFER'}")
        ok &= same
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(SCRIPTS)))
