"""Next-byte pretraining of the paper's architecture on DCLM text.

    python train.py --size 1M --seed 0 --tokens 4e9 --lr 1e-3 \\
        --ckpt-tokens 0.1e9,0.25e9,0.5e9,1e9,2e9,4e9

Arms:
  * ``dclm``: from a GPT-2-style random init.
  * ``sp2dclm``: warm start from a self-play checkpoint (``--init-from``),
    as in the paper's Figure 6.

Protocol (Figure 6, adapted): AdamW(0.9, 0.95), weight decay on all
parameters, 64 x 4096-byte sequences per step (gradient accumulation over
``--micro-bs``), linear warmup over 2% of the run, then constant LR so a
checkpoint at any token count is a plain mid-run model. ``--cooldown-steps``
adds a final cosine decay to zero. bf16 autocast.

Sequences are ``O`` + 4095 bytes, drawn without replacement from non-overlapping
windows of the part files (reshuffled each epoch), matching the scorer's
input convention. Checkpoints use the shared layout
(``<out>/<arm>/<size>/seed-<seed>/learner_<tokens in M>M.pth``, e.g. ``learner_250M.pth``), so ``icl.run``
and the scorer read them like the Hugging Face ones. ``resume.pt`` is
refreshed every ``--resume-every`` seconds and picked up automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import (CONTEXT, N_NONEMB, PREFIX, ROOT, arch_config, build_model,
                    hidden_states, init_weights, load_learner, save_learner, wandb_init)

WINDOW = CONTEXT - 1   # data bytes per sequence; 'O' fills position 0


class ByteWindows:
    """Non-overlapping WINDOW-byte slices over memmapped part files."""

    def __init__(self, data_dir: Path, seed: int):
        parts = sorted(Path(data_dir).glob("part_*.bin"))
        if not parts:
            raise SystemExit(f"no part_*.bin in {data_dir}; run prepare_dclm_train.py")
        self.maps = [np.memmap(p, dtype=np.uint8, mode="r") for p in parts]
        counts = [len(m) // WINDOW for m in self.maps]
        self.part_of = np.repeat(np.arange(len(parts)), counts)
        self.offset = np.concatenate([np.arange(c) * WINDOW for c in counts])
        self.n = len(self.offset)
        self.bytes = self.n * WINDOW
        self.rng = np.random.default_rng(seed)
        self.epoch, self.cursor = 0, 0
        self.perm = self.rng.permutation(self.n)

    def batch(self, n: int) -> np.ndarray:
        out = np.empty((n, CONTEXT), dtype=np.int64)
        out[:, 0] = PREFIX
        for r in range(n):
            if self.cursor == self.n:
                self.epoch += 1
                self.cursor = 0
                self.perm = self.rng.permutation(self.n)
            i = self.perm[self.cursor]
            self.cursor += 1
            o = self.offset[i]
            out[r, 1:] = self.maps[self.part_of[i]][o:o + WINDOW]
        return out


class Prefetcher:
    """Background thread producing pinned CPU batches."""

    def __init__(self, src: ByteWindows, n: int, depth: int = 4):
        self.q: queue.Queue = queue.Queue(depth)
        self.src, self.n = src, n
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            with self.lock:
                b = torch.from_numpy(self.src.batch(self.n))
            self.q.put(b.pin_memory() if torch.cuda.is_available() else b)

    def get(self):
        return self.q.get()


class LossModule(nn.Module):
    """Wrapper so ``torch.compile`` sees the whole loss computation."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, seq):
        h = hidden_states(self.model, seq[:, :-1])
        logits = self.model.lm_head(h).float()
        return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                           seq[:, 1:].reshape(-1))


def load_val(path: Path, n: int) -> torch.Tensor:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line)["sequence"])
            if len(rows) >= n:
                break
    a = np.asarray(rows, dtype=np.int64)
    assert a.shape[1] == WINDOW, f"val corpus window {a.shape[1]} != {WINDOW}"
    return torch.from_numpy(np.concatenate([np.full((len(a), 1), PREFIX), a], 1))


@torch.no_grad()
def val_bpb(lossmod, val, device, micro_bs) -> float:
    lossmod.eval()
    tot = 0.0
    for lo in range(0, len(val), micro_bs):
        b = val[lo:lo + micro_bs].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            tot += lossmod(b).item() * len(b)
    lossmod.train()
    return tot / len(val) / math.log(2)


def lr_at(step, total, warmup, cooldown, lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    if cooldown and step >= total - cooldown:
        t = (step - (total - cooldown)) / cooldown
        return lr * 0.5 * (1 + math.cos(math.pi * t))
    return lr


def make_optimizer(model, lr, wd, device):
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8,
                             weight_decay=wd, fused=device.type == "cuda")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default=None, help="default: dclm, or sp2dclm with --init-from")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="self-play checkpoint to warm start from")
    ap.add_argument("--data", type=Path, default=ROOT / "data/dclm_train")
    ap.add_argument("--val", type=Path,
                    default=ROOT.parent / "self_play_pretraining/scoring/data/c4096/dclm.jsonl")
    ap.add_argument("--val-seqs", type=int, default=64)
    ap.add_argument("--tokens", type=float, required=True, help="total training tokens")
    ap.add_argument("--ckpt-tokens", default="", help="comma list of token counts to save at")
    ap.add_argument("--batch-seqs", type=int, default=64)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--cooldown-steps", type=int, default=0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--resume-every", type=float, default=900, help="seconds")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "ckpts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                    help="log to Weights & Biases (needs WANDB_API_KEY, e.g. in ../.env)")
    ap.add_argument("--wandb-group", default=None, help="default: the arm")
    args = ap.parse_args()

    device = torch.device(args.device)
    arm = args.arm or ("sp2dclm" if args.init_from else "dclm")
    run_dir = args.out / arm / args.size / f"seed-{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.init_from:
        model = load_learner(args.init_from, "cpu")
        if model.get_config() != arch_config(args.size):
            raise SystemExit(f"--init-from is {model.get_config()}, not size {args.size}")
    else:
        model = build_model(arch_config(args.size))
        init_weights(model)
    model.to(device).train()
    lossmod = LossModule(model)
    step_fn = torch.compile(lossmod) if args.compile else lossmod
    opt = make_optimizer(model, args.lr, args.wd, device)

    tokens_per_step = args.batch_seqs * CONTEXT
    total_steps = math.ceil(args.tokens / tokens_per_step)
    warmup = max(1, int(args.warmup_frac * total_steps))
    accum = args.batch_seqs // args.micro_bs
    assert accum * args.micro_bs == args.batch_seqs, "--micro-bs must divide --batch-seqs"
    ckpt_at = sorted(int(float(t)) for t in args.ckpt_tokens.split(",") if t) + [int(args.tokens)]

    data = ByteWindows(args.data, args.seed)
    val = load_val(args.val, args.val_seqs) if args.val.is_file() else None
    if val is None:
        print(f"[warn] no validation corpus at {args.val}; val BPB disabled")

    step, log_path = 0, run_dir / "log.jsonl"
    resume = run_dir / "resume.pt"
    if resume.exists():
        st = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        data.epoch, data.cursor = st["data"]["epoch"], st["data"]["cursor"]
        data.rng.bit_generator.state = st["data"]["rng"]
        data.perm = st["data"]["perm"]
        step = st["step"]
        ckpt_at = [t for t in ckpt_at if t > step * tokens_per_step]
        print(f"resumed from step {step} ({step * tokens_per_step / 1e9:.2f}B tokens)")
    else:
        (run_dir / "run.json").write_text(json.dumps(
            {**{k: str(v) for k, v in vars(args).items()}, "arm": arm,
             "N_nonemb": N_NONEMB.get(args.size), "total_steps": total_steps,
             "warmup_steps": warmup, "data_bytes": data.bytes,
             "params": sum(p.numel() for p in model.parameters())}, indent=1))
    wb = wandb_init(args.wandb, run_dir / "wandb_id",
                    name=f"{arm}-{args.size}-s{args.seed}", group=args.wandb_group or arm,
                    job_type="train", tags=[arm, args.size],
                    config={**json.loads((run_dir / "run.json").read_text()),
                            "run_dir": str(run_dir)})
    if wb:
        # x-axis = tokens seen, so a resumed run lines up with its first part
        wb.define_metric("tokens_B")
        for pat in ("train/*", "val/*"):
            wb.define_metric(pat, step_metric="tokens_B")

    print(f"{arm}/{args.size} seed {args.seed}: {total_steps} steps x {tokens_per_step} tok "
          f"(accum {accum} x {args.micro_bs}), warmup {warmup}, data "
          f"{data.bytes / 1e9:.2f} GB ({args.tokens / data.bytes:.2f} epochs)", flush=True)
    pre = Prefetcher(data, args.micro_bs)
    t_last_resume = t_log = time.time()
    tok_since_log = 0
    with open(log_path, "a") as logf:
        while step < total_steps:
            lr = lr_at(step, total_steps, warmup, args.cooldown_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_acc = 0.0
            for _ in range(accum):
                b = pre.get().to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    loss = step_fn(b) / accum
                loss.backward()
                loss_acc += loss.detach()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tokens = step * tokens_per_step
            tok_since_log += tokens_per_step

            rec = None
            if step % args.log_every == 0 or step == total_steps:
                dt = time.time() - t_log
                rec = {"step": step, "tokens": tokens, "loss": float(loss_acc),
                       "bpb": float(loss_acc) / math.log(2), "lr": lr,
                       "gnorm": float(gnorm), "tok_s": tok_since_log / dt,
                       "epoch": data.epoch, "time": time.time()}
                t_log, tok_since_log = time.time(), 0
            if val is not None and (step % args.eval_every == 0 or step == total_steps):
                rec = rec or {"step": step, "tokens": tokens, "time": time.time()}
                rec["val_bpb"] = val_bpb(lossmod, val, device, args.micro_bs)
            if rec:
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                if wb:
                    wb.log({"tokens_B": tokens / 1e9,
                            **{("val/bpb" if k == "val_bpb" else f"train/{k}"): v
                               for k, v in rec.items() if k not in ("tokens", "time")}})
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in rec.items() if k != "time"), flush=True)
            while ckpt_at and tokens >= ckpt_at[0]:
                t = ckpt_at.pop(0)
                path = run_dir / f"learner_{t / 1e6:g}M.pth"
                save_learner(model, path, round=step, tokens=tokens, arm=arm,
                             init_from=str(args.init_from or ""))
                print(f"saved {path}", flush=True)
                if wb:
                    wb.summary["last_checkpoint"] = path.name
            if time.time() - t_last_resume > args.resume_every or step == total_steps:
                with pre.lock:
                    dstate = {"epoch": data.epoch, "cursor": data.cursor,
                              "rng": data.rng.bit_generator.state, "perm": data.perm}
                # the prefetch queue holds a few batches already drawn; replaying
                # from this cursor skips them, which is harmless
                tmp = resume.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "data": dstate}, tmp)
                tmp.replace(resume)
                t_last_resume = time.time()
    (run_dir / "done").write_text(str(step * tokens_per_step))
    if wb:
        wb.finish()
    print("done")


if __name__ == "__main__":
    main()
