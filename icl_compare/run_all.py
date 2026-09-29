"""The full experiment, end to end, on one GPU pod. Safe to re-run: every stage
skips work that is already done, and training resumes from ``resume.pt``.

    nohup python3 run_all.py --price 1.23 --auto-terminate > ~/run_all.log 2>&1 &

Stages
  1. data     DCLM training bytes, the DCLM validation bake, and the released
              self-play (sp) and universal-prior (up) checkpoints.
  2. lr       per-size LR sweep: three short DCLM runs, best validation BPB wins.
  3. eval-sp  ICL of the sp and up checkpoints (all sizes, 4 seeds).
  4. dclm     DCLM from scratch, 100k ... 6M, to the paper's ICL point
              (17.7B tokens), with checkpoints at every matched self-play
              token count; then ICL.
  5. warm     self-play (round 8191) -> DCLM warm starts, 0.5B tokens; then ICL.
  6. dclm24   DCLM 24M to 17.7B tokens, last because it is the most expensive.
  7. eval-words  the word-level ICL suite on every checkpoint (all arms/sizes).
  8. extra    extra DCLM seeds, in priority order, each only if the wallet can
              afford it (estimated from measured speed x price) with a reserve
              left for the report and upload; each is then scored.
  9. report   analysis/report.py: m-curves, seed CIs, per-seed checks, figures.
 10. final    upload every DCLM checkpoint, log and result to W&B, write
              ALL_DONE, and (with --auto-terminate) delete this pod.

Safety for an unattended run
  * A failed step is retried twice (training resumes, finished evals are skipped).
  * If it still fails, or the wallet drops below --min-balance (the lower of
    the Prime API balance and start balance minus elapsed hours x --price),
    the run stops, uploads everything it has to W&B ("salvage"), and then
    deletes the pod if --auto-terminate is set.
  * The pod is only deleted after the W&B upload has succeeded.

Everything logs to W&B (``../.env``); progress also goes to ``state/progress.log``.
``--smoke`` runs the same pipeline on CPU in minutes (100k/500k, tiny token
counts, separate dirs and W&B project) to check the plumbing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from common import MATCH_TOKENS, ROOT, SP_ROUNDS, load_env

PY = sys.executable
T0 = time.time()
PRIME_API = "https://api.primeintellect.ai/api/v1"


class Cfg:
    """Pipeline settings; ``smoke()`` shrinks everything for a CPU test."""
    state = ROOT / "state"
    ckpts, sweep_dir, results = "ckpts", "ckpts_sweep", "results/icl"
    data = "data/dclm_train"
    small = ["100k", "500k", "1M", "3M", "6M"]
    big = "24M"
    up_sizes = ["100k", "500k", "1M", "3M", "6M"]   # the universal-prior ladder stops at 6M
    match = list(MATCH_TOKENS)                       # 1.6B ... 17.7B (self-play rounds 256 ... 2816)
    early = [int(1e8), int(2.5e8), int(5e8), int(1e9)]
    lr_grid = {"100k": [1e-3, 2e-3, 4e-3], "500k": [1e-3, 2e-3, 4e-3],
               "1M": [1e-3, 2e-3, 4e-3], "3M": [1e-3, 2e-3, 4e-3],
               "6M": [5e-4, 1e-3, 2e-3], "24M": [5e-4, 1e-3, 2e-3]}
    lr_tokens = {"24M": 3e8}                         # others: default below
    lr_tokens_default = 5e8
    warm_lr, warm_tokens = 3e-3, 5e8                 # the paper's warm-start LR (Fig. 6)
    train_extra: list = ["--compile"]
    eval_extra: list = ["--bf16"]
    eval_every, lr_eval_every = 500, 100
    sp_seeds = 4
    words_results = "results/icl_words"
    # extra DCLM seeds, highest value first: (size, seed). Overridable at any
    # time by writing state/extra_jobs.json ([["6M", 1], ...]); it is re-read
    # before each job.
    extra_jobs = [("6M", 1), ("3M", 1), ("1M", 1), ("500k", 1),
                  ("1M", 2), ("3M", 2), ("6M", 2), ("500k", 2)]
    extra_reserve = 1.0                              # $ above --min-balance kept for report/upload
    retries = 2
    watch_every = 300                                # seconds between wallet checks

    @property
    def all(self):
        return self.small + [self.big]

    def smoke(self):
        self.state = ROOT / "state_smoke"
        self.ckpts, self.sweep_dir, self.results = ("ckpts_smoke", "ckpts_sweep_smoke",
                                                    "results/icl_smoke")
        self.small, self.big, self.up_sizes = ["100k"], "500k", ["100k"]
        self.match = [int(1e5), int(2e5)]
        self.early = [int(5e4)]
        self.lr_grid = {"100k": [1e-3, 4e-3], "500k": [1e-3, 4e-3]}
        self.lr_tokens, self.lr_tokens_default = {}, 1e5
        self.warm_tokens = 1e5
        self.train_extra = ["--batch-seqs", 4, "--micro-bs", 2, "--val-seqs", 2]
        self.eval_extra = ["--max-len", 40, "--device", "cpu"]
        self.eval_every, self.lr_eval_every = 3, 3
        self.sp_seeds = 2
        self.words_results = "results/icl_words_smoke"
        self.extra_jobs = [("100k", 1), ("500k", 1)]
        self.watch_every = 15
        os.environ["WANDB_PROJECT"] = "icl-selfplay-test"


C = Cfg()


class Stop(Exception):
    """Raised to abandon the pipeline (low balance or a step that keeps failing)."""


LOW_BALANCE = threading.Event()
CHILD: list = [None]                                 # the running subprocess, for the watchdog


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')} +{(time.time() - T0) / 3600:.2f}h] {msg}"
    print(line, flush=True)
    with open(C.state / "progress.log", "a") as f:
        f.write(line + "\n")


def sh(*cmd, cwd=ROOT) -> None:
    """Run a step, retrying failures; raise Stop on low balance or repeated failure."""
    for attempt in range(C.retries + 1):
        if LOW_BALANCE.is_set():
            raise Stop("wallet below the safety threshold")
        log(("$ " if attempt == 0 else f"retry {attempt}: $ ") + " ".join(map(str, cmd)))
        CHILD[0] = subprocess.Popen([str(c) for c in cmd], cwd=cwd)
        rc = CHILD[0].wait()
        CHILD[0] = None
        if rc == 0:
            return
        if LOW_BALANCE.is_set():
            raise Stop("wallet below the safety threshold")
        log(f"step failed with exit code {rc}")
        time.sleep(30 if attempt < C.retries else 0)
    raise Stop(f"step failed {C.retries + 1} times: {' '.join(map(str, cmd))}")


def optional(what: str, fn, *a) -> None:
    """Run a step whose failure should not end the pipeline (low balance still does)."""
    try:
        fn(*a)
    except Stop as e:
        if LOW_BALANCE.is_set():
            raise
        log(f"{what} FAILED ({e}); continuing without it")


def stage(name: str):
    """Decorator: run once; a marker file records completion."""
    def wrap(fn):
        def run(*a, **k):
            marker = C.state / f"{name}.done"
            if marker.exists():
                log(f"stage {name}: already done")
                return
            log(f"stage {name}: start")
            fn(*a, **k)
            marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            log(f"stage {name}: done")
        return run
    return wrap


# ------------------------------------------------------------------ wallet watchdog

def prime_balance() -> float | None:
    key = os.environ.get("PRIME_API_KEY")
    if not key:
        return None
    try:
        req = urllib.request.Request(f"{PRIME_API}/billing/wallet",
                                     headers={"Authorization": f"Bearer {key}"})
        return float(json.load(urllib.request.urlopen(req, timeout=30))["balance_usd"])
    except Exception as e:                           # a flaky API must not stop the run
        log(f"wallet check failed: {e}")
        return None


def current_balance(price: float) -> float | None:
    """Lower of the API balance and start balance - elapsed hours x price."""
    start_file = C.state / "start_balance"
    if not start_file.exists():
        return None
    est = float(start_file.read_text()) - (time.time() - T0) / 3600 * price
    api = prime_balance()
    return min(x for x in (api, est) if x is not None)


def watchdog(price: float, min_balance: float) -> None:
    """Stop the run before the wallet runs out.

    Uses the lower of the API balance and ``start - elapsed_hours * price``, in
    case the API only reflects charges once a pod is billed.
    """
    start_file = C.state / "start_balance"
    if not start_file.exists():
        b = prime_balance()
        if b is None:
            log("watchdog: no wallet access (PRIME_API_KEY unset); low-balance stop disabled")
            return
        start_file.write_text(str(b))
    start = float(start_file.read_text())
    log(f"watchdog: start balance ${start:.2f}, price ${price}/h, stop below ${min_balance}")
    while not LOW_BALANCE.is_set():
        api = prime_balance()
        est = start - (time.time() - T0) / 3600 * price
        bal = min(x for x in (api, est) if x is not None)
        if bal < min_balance:
            log(f"watchdog: balance ${bal:.2f} (api {api}, estimate ${est:.2f}) < "
                f"${min_balance}: stopping")
            LOW_BALANCE.set()
            if CHILD[0] is not None:
                CHILD[0].terminate()
            return
        time.sleep(C.watch_every)


# ------------------------------------------------------------------ helpers

def micro_bs(size: str) -> list:
    if "--micro-bs" in C.train_extra:
        return []
    import torch
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    return ["--micro-bs", 64 if (size != "24M" or mem > 60) else 32]


def train(size, arm, lr, tokens, ckpts, init_from=None, group=None, out=None, eval_every=None,
          seed=0):
    cmd = [PY, "train.py", "--size", size, "--seed", seed, "--arm", arm, "--lr", lr,
           "--tokens", int(tokens), "--ckpt-tokens", ",".join(str(int(t)) for t in ckpts),
           "--eval-every", eval_every or C.eval_every, "--log-every", 50,
           "--data", C.data, "--out", out or C.ckpts, *micro_bs(size), *C.train_extra]
    if init_from:
        cmd += ["--init-from", init_from]
    if group:
        cmd += ["--wandb-group", group]
    sh(*cmd)


def evaluate(arm, size, n_seeds, *extra):
    sh(PY, "-m", "icl.evaluate_arm", "--arm", arm, "--size", size, "--n-seeds", n_seeds,
       "--ckpt-root", C.ckpts, "--out-root", C.results, *C.eval_extra, *extra)


def val_rows(run_dir: Path) -> list:
    return [json.loads(l) for l in open(run_dir / "log.jsonl")]


def last_val_bpb(run_dir: Path, n=2) -> float:
    vals = [r["val_bpb"] for r in val_rows(run_dir) if "val_bpb" in r]
    return sum(vals[-n:]) / len(vals[-n:])


def tok_per_s(run_dir: Path) -> float:
    r = [r["tok_s"] for r in val_rows(run_dir) if "tok_s" in r]
    r = sorted(r[len(r) // 2:])                 # skip compile warmup, take the median
    return r[len(r) // 2]


# ------------------------------------------------------------------ stages

@stage("data")
def s_data(gbytes: float):
    if not C.data.startswith("data/"):
        log(f"smoke: using existing data at {C.data}")
    else:
        sh(PY, "prepare_dclm_train.py", "--out", C.data, "--gbytes", gbytes,
           "--workers", min(16, os.cpu_count() or 4))
    val = ROOT.parent / "self_play_pretraining/scoring/data/c4096/dclm.jsonl"
    if not val.exists():
        sh(PY, "-m", "src.scripts.prepare_dclm_benchmark", "--context-length", 4096,
           "--num-sequences", 2048, cwd=ROOT.parent / "self_play_pretraining/scoring")
    rounds = ",".join(map(str, SP_ROUNDS))
    sh(PY, "fetch_ckpts.py", "--arm", "sp", "--sizes", ",".join(C.all), "--rounds", rounds,
       "--n-seeds", C.sp_seeds, "--out", C.ckpts)
    sh(PY, "fetch_ckpts.py", "--arm", "up", "--sizes", ",".join(C.up_sizes), "--rounds",
       rounds, "--n-seeds", C.sp_seeds, "--out", C.ckpts)


@stage("lr")
def s_lr():
    choice = {}
    for size in C.all:
        best = None
        for lr in C.lr_grid[size]:
            arm = f"lrsweep-{lr:g}"
            run_dir = ROOT / C.sweep_dir / arm / size / "seed-0"
            if not (run_dir / "done").exists():
                train(size, arm, lr, C.lr_tokens.get(size, C.lr_tokens_default), [],
                      group="lrsweep", out=C.sweep_dir, eval_every=C.lr_eval_every)
            bpb = last_val_bpb(run_dir)
            log(f"lr sweep {size} lr={lr:g}: val_bpb {bpb:.4f}, "
                f"{tok_per_s(run_dir) / 1e6:.3f}M tok/s")
            if best is None or bpb < best[1]:
                best = (lr, bpb)
        choice[size] = best[0]
        log(f"lr sweep {size}: chose {best[0]:g}")
    (C.state / "lr.json").write_text(json.dumps(choice, indent=1))


@stage("eval-sp")
def s_eval_sp():
    for size in C.all:
        evaluate("sp", size, C.sp_seeds)
    for size in C.up_sizes:
        evaluate("up", size, C.sp_seeds)


def dclm_ckpts(tokens: int) -> list:
    return sorted({t for t in C.early + C.match if t < tokens} | {tokens})


def dclm_run(size: str, tokens: int):
    lr = json.loads((C.state / "lr.json").read_text())[size]
    train(size, "dclm", lr, tokens, dclm_ckpts(tokens))
    evaluate("dclm", size, 1)


@stage("dclm")
def s_dclm():
    for size in C.small:
        dclm_run(size, C.match[-1])


@stage("warm")
def s_warm():
    wt = C.warm_tokens
    for size in C.all:
        init = sorted((ROOT / C.ckpts / "sp" / size).glob("seed-*/learner_8191.pth"))[0]
        train(size, "sp2dclm", C.warm_lr, wt, [int(wt / 10), int(wt / 5), int(wt / 2)],
              init_from=init)
        evaluate("sp2dclm", size, 1)


@stage("dclm24")
def s_dclm_big():
    dclm_run(C.big, C.match[-1])


@stage("eval-words")
def s_eval_words():
    runs = ([("sp", s, C.sp_seeds) for s in C.all] + [("up", s, C.sp_seeds) for s in C.up_sizes]
            + [("dclm", s, 1) for s in C.all] + [("sp2dclm", s, 1) for s in C.all])
    for arm, size, n in runs:
        optional(f"word eval {arm}/{size}", evaluate, arm, size, n, "--suite", "words",
                 "--out-root", C.words_results, "--run-name", f"iclw-{arm}-{size}")


@stage("extra")
def s_extra(price: float, min_balance: float):
    """Extra DCLM seeds while the wallet allows, highest priority first."""
    lr = json.loads((C.state / "lr.json").read_text())
    tokens = C.match[-1]
    tried: set = set()
    while True:
        queue_file = C.state / "extra_jobs.json"
        try:
            queue = [tuple(j) for j in json.loads(queue_file.read_text())]
        except (OSError, ValueError):
            queue = C.extra_jobs
        pending = [(sz, sd) for sz, sd in queue if (sz, int(sd)) not in tried]
        if not pending:
            break
        size, seed = pending[0][0], int(pending[0][1])
        tried.add((size, seed))
        run_dir = ROOT / C.ckpts / "dclm" / size / f"seed-{seed}"
        out_root = f"{C.results}_s{seed}"
        scored = (ROOT / out_root / "dclm" / size / f"{tokens / 1e6:g}M.json").exists()
        if (run_dir / "done").exists() and scored:
            log(f"extra {size} seed {seed}: already done")
            continue
        done_tok = 0
        if (run_dir / "log.jsonl").exists():
            done_tok = max((r["tokens"] for r in val_rows(run_dir)), default=0)
        rate = tok_per_s(sorted((ROOT / C.sweep_dir).glob(f"lrsweep-*/{size}/seed-0"))[0])
        cost = ((tokens - done_tok) / rate / 3600 + 0.15) * price
        bal = current_balance(price)
        if bal is not None and bal - cost < min_balance + C.extra_reserve:
            log(f"extra {size} seed {seed}: skipped, needs ~${cost:.2f} but balance is "
                f"~${bal:.2f} (keeping ${min_balance + C.extra_reserve:.2f})")
            continue
        log(f"extra {size} seed {seed}: ~${cost:.2f}, balance ~${bal if bal is not None else float('nan'):.2f}")
        def job():
            train(size, "dclm", lr[size], tokens, dclm_ckpts(tokens), seed=seed)
            evaluate("dclm", size, 1, "--seed-glob", f"seed-{seed}", "--suite",
                     "fig4,printable,words", "--out-root", out_root,
                     "--run-name", f"icl-dclm-{size}-s{seed}")
        optional(f"extra {size} seed {seed}", job)


@stage("report")
def s_report():
    smoke = ["--smoke"] if C.ckpts.endswith("smoke") else []
    optional("report", sh, PY, "analysis/report.py", "--results", "results", "--out",
             "results/report" + ("_smoke" if smoke else ""), "--ckpts", C.ckpts, "--wandb", *smoke)


def upload_all(name: str, console_log: Path | None) -> bool:
    """Upload every DCLM checkpoint dir, sweep log, ICL result and log to W&B.

    Returns True only if every artifact finished uploading. Retries three times.
    """
    import wandb
    for attempt in range(3):
        try:
            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "icl-selfplay"),
                             name=name, job_type="upload", reinit="finish_previous")
            for arm in ("dclm", "sp2dclm"):
                for size_dir in sorted((ROOT / C.ckpts / arm).glob("*")):
                    art = wandb.Artifact(f"ckpts-{arm}-{size_dir.name}", type="model")
                    art.add_dir(str(size_dir))
                    run.log_artifact(art).wait()
                    log(f"uploaded checkpoints {arm}/{size_dir.name}")
            art = wandb.Artifact("results-all", type="results")
            art.add_dir(str(ROOT / "results"), name="results")
            for p in sorted((ROOT / C.sweep_dir).glob("*/*/seed-0/*")):
                if p.name in ("log.jsonl", "run.json"):
                    art.add_file(str(p), name=f"lrsweep/{p.relative_to(ROOT / C.sweep_dir)}")
            for p in sorted(C.state.glob("*")):
                if p.is_file() and p.name != "pod.env":      # never upload the Prime API key
                    art.add_file(str(p), name=f"state/{p.name}")
            if console_log and console_log.exists():
                art.add_file(str(console_log), name="run_all.log")
            run.log_artifact(art).wait()
            run.finish()
            log("uploaded results, logs and state")
            return True
        except Exception as e:
            log(f"upload attempt {attempt + 1} failed: {e}")
            time.sleep(60)
    return False


def terminate_pod():
    pod, key = os.environ.get("POD_ID"), os.environ.get("PRIME_API_KEY")
    if not (pod and key):
        log("auto-terminate: POD_ID/PRIME_API_KEY not set; leaving the pod running")
        return
    log(f"auto-terminate: deleting pod {pod}")
    req = urllib.request.Request(f"{PRIME_API}/pods/{pod}",
                                 headers={"Authorization": f"Bearer {key}"}, method="DELETE")
    urllib.request.urlopen(req, timeout=60)


def main():
    global T0
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--price", type=float, default=0.0,
                    help="$/h of this pod, for the spend estimate in the wallet watchdog")
    ap.add_argument("--min-balance", type=float, default=2.0,
                    help="stop, upload and terminate if the wallet falls below this")
    ap.add_argument("--data-gbytes", type=float, default=18.5)
    ap.add_argument("--auto-terminate", action="store_true",
                    help="delete the pod after the W&B upload succeeds (needs POD_ID and "
                         "PRIME_API_KEY, which launch.sh writes to state/pod.env)")
    ap.add_argument("--console-log", type=Path, default=Path.home() / "run_all.log",
                    help="this script's stdout file, uploaded at the end")
    ap.add_argument("--smoke", type=Path, default=None, metavar="DATA_DIR",
                    help="CPU plumbing test using the given small part_*.bin directory")
    args = ap.parse_args()
    if args.smoke:
        C.smoke()
        C.data = str(args.smoke)
    C.state.mkdir(exist_ok=True)
    load_env()
    env_file = C.state / "pod.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)
    started = C.state / "started_at"            # elapsed time counts from the first launch
    if started.exists():
        T0 = float(started.read_text())
    else:
        started.write_text(str(T0))
    log(f"run_all: price ${args.price}/h, min balance ${args.min_balance}, "
        f"auto-terminate={args.auto_terminate}, smoke={bool(args.smoke)}")
    threading.Thread(target=watchdog, args=(args.price, args.min_balance), daemon=True).start()

    try:
        s_data(args.data_gbytes)
        s_lr()
        s_eval_sp()
        s_dclm()
        s_warm()
        s_dclm_big()
        s_eval_words()
        s_extra(args.price, args.min_balance)
        s_report()
        uploaded = upload_all("final-upload", args.console_log)
        if uploaded:
            (C.state / "ALL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            log("ALL DONE")
    except Stop as e:
        log(f"STOPPED: {e}. Uploading what exists (salvage).")
        uploaded = upload_all("salvage-upload", args.console_log)
    except Exception as e:                      # a bug in this script: still save the work
        log(f"STOPPED on unexpected error {e!r}. Uploading what exists (salvage).")
        uploaded = upload_all("salvage-upload", args.console_log)

    if not uploaded:
        log("W&B upload FAILED; NOT terminating so nothing is lost. Check the pod.")
        sys.exit(1)
    if args.auto_terminate:
        terminate_pod()


if __name__ == "__main__":
    main()
