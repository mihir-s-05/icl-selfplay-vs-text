"""Training-throughput benchmark: tokens/s per model size on this GPU.

    python bench.py --sizes 1M,3M,6M,24M --compile --price 1.20

For each size, tries each micro-batch (largest first, skipping on OOM), runs
full 64-sequence optimizer steps on random bytes (no data loading, so this is
the model's ceiling), and reports tokens/s, peak memory, and cost per billion
tokens at ``--price`` $/h. Results are appended to ``--out`` as JSON lines.

Concurrency test: launch one process per size with a single ``--micro-bs``,
the same ``--start-at`` (a unix time after everyone's compile finishes) and
``--duration``; every process then times the same window while sharing the GPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common import CONTEXT, PREFIX, ROOT, arch_config, build_model, init_weights
from train import LossModule, make_optimizer


def run(size, micro_bs, batch_seqs, steps, warmup, compile_, device,
        start_at=None, duration=None):
    model = build_model(arch_config(size))
    init_weights(model)
    model.to(device).train()
    lossmod = LossModule(model)
    fn = torch.compile(lossmod) if compile_ else lossmod
    opt = make_optimizer(model, 1e-3, 0.1, device)
    accum = batch_seqs // micro_bs
    data = torch.randint(0, 256, (micro_bs, CONTEXT), device=device)
    data[:, 0] = PREFIX
    torch.cuda.reset_peak_memory_stats()

    def step():
        for _ in range(accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = fn(data) / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    t_c = time.time()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t_c = time.time() - t_c
    if start_at is not None:                  # concurrent mode: common start time
        time.sleep(max(0.0, start_at - time.time()))
    t0 = time.time()
    if duration is None:
        for _ in range(steps):
            step()
    else:                                     # count whole steps inside the window
        steps = 0
        while time.time() - t0 < duration:
            step()
            steps += 1
            if steps % 4 == 0:
                torch.cuda.synchronize()
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {"tok_s": steps * batch_seqs * CONTEXT / dt, "step_s": dt / steps,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "warmup_s": t_c}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="1M,3M,6M,24M")
    ap.add_argument("--micro-bs", default="64,32,16,8")
    ap.add_argument("--batch-seqs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--price", type=float, default=None, help="$/h of this GPU")
    ap.add_argument("--start-at", type=float, default=None,
                    help="unix time to start timing (after compile warmup); lets "
                         "several concurrent processes share one timing window")
    ap.add_argument("--duration", type=float, default=None,
                    help="time the run for this many seconds instead of --steps")
    ap.add_argument("--tag", default="", help="free-text label stored with results")
    ap.add_argument("--out", type=Path, default=ROOT / "results/bench.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {gpu}  torch {torch.__version__}  compile={args.compile}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes.split(","):
        best = None
        for mb in (int(x) for x in args.micro_bs.split(",")):
            if args.batch_seqs % mb:
                continue
            try:
                r = run(size, mb, args.batch_seqs, args.steps, args.warmup, args.compile,
                        device, args.start_at, args.duration)
            except torch.OutOfMemoryError:
                print(f"{size:>4} micro_bs={mb:<3} OOM", flush=True)
                torch.cuda.empty_cache()
                continue
            torch.cuda.empty_cache()
            rec = {"gpu": gpu, "size": size, "micro_bs": mb, "compile": args.compile,
                   "tag": args.tag, **r}
            if args.price:
                rec["usd_per_Btok"] = args.price / (r["tok_s"] * 3600 / 1e9)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{size:>4} micro_bs={mb:<3} {r['tok_s'] / 1e3:8.1f}k tok/s  "
                  f"{r['peak_gb']:5.1f} GB  step {r['step_s']:.3f}s  "
                  f"1B tok = {1e9 / r['tok_s'] / 3600:5.2f} h"
                  + (f"  ${rec['usd_per_Btok']:.2f}/Btok" if args.price else ""), flush=True)
            if best is None or r["tok_s"] > best[1]:
                best = (mb, r["tok_s"])
        if best:
            print(f"{size:>4} BEST micro_bs={best[0]}: {best[1] / 1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
