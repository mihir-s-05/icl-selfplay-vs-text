"""Run the ICL suites on a set of checkpoints (one arm/size/round, any seeds).

    python -m icl.run --ckpts "ckpts/sp/24M/seed-*/learner_2816.pth" \\
        --suite fig4 --out results/sp/24M/r2816.json

Every checkpoint is scored separately. The next-byte distribution at each
scored position is kept per seed, so the output has both single-seed metrics
(the primary comparison across arms) and the probability ensemble over all
given seeds (the paper's convention: average probabilities, then argmax).

``--max-len`` skips cells whose longest prompt exceeds it. The prompts of the
remaining cells are unchanged, so a CPU run can check the short cells against
the paper exactly. ``--probs`` also saves the per-seed probabilities (fp16).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import hidden_states, load_learner  # noqa: E402
from icl import tasks as T  # noqa: E402


@torch.inference_mode()
def cell_probs(model, cell: T.Cell, device, max_tokens: int, amp: bool) -> np.ndarray:
    """``[N, P, 256]`` float32 probabilities at the scored positions.

    Trials are grouped by exact length (no padding: a pad before the ``O``
    prefix would be off-distribution) and chunked to ``max_tokens`` per forward.
    P is the longest scored-position list; shorter rows are zero-padded.
    """
    N = len(cell.seqs)
    P = max(len(s) for s in cell.score_at)
    out = np.zeros((N, P, 256), dtype=np.float32)
    by_len: dict[int, list[int]] = {}
    for i, s in enumerate(cell.seqs):
        by_len.setdefault(len(s), []).append(i)
    for L, idx in by_len.items():
        bs = max(1, max_tokens // L)
        for lo in range(0, len(idx), bs):
            chunk = idx[lo:lo + bs]
            X = torch.tensor([cell.seqs[i] for i in chunk], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                h = hidden_states(model, X)
            for r, i in enumerate(chunk):
                pos = torch.tensor(cell.score_at[i], device=device)
                logits = model.lm_head(h[r, pos].float())
                out[i, :len(pos)] = torch.softmax(logits.float(), -1).cpu().numpy()
    return out


def metrics(cell: T.Cell, p: np.ndarray) -> dict:
    """Metrics for one probability array ``[N, P, 256]`` (a seed or an ensemble)."""
    N = len(cell.targets)
    if cell.section == "control":
        n = cell.extra["n_normal"]
        pn, pd = p[:n], p[n:]
        tg = np.array([t[0] for t in cell.targets[:n]])
        pred = pn[:, 0].argmax(-1)
        Q = cell.extra["queries"]
        return dict(
            acc=float((pred == tg).mean()),
            acc_decoupled=float((pd[:, 0].argmax(-1) == tg).mean()),
            pred_is_other_query_byte=float(np.mean(
                [(pred[i] in Q[i]) and pred[i] != tg[i] for i in range(n)])),
            acc_within_4=float((np.abs(pred.astype(int) - tg) <= 4).mean()),
            mae=float(np.abs(pred.astype(int) - tg).mean()))
    correct, pcorr, n_pos = [], [], []
    P = p.shape[1]
    pos_ok = np.zeros((N, P), dtype=bool)
    pos_valid = np.zeros((N, P), dtype=bool)
    for i, tg in enumerate(cell.targets):
        k = len(tg)
        pred = p[i, :k].argmax(-1)
        pos_ok[i, :k] = pred == np.asarray(tg)
        pos_valid[i, :k] = True
        correct.append(bool(pos_ok[i, :k].all()))
        pcorr.append(float(p[i, np.arange(k), tg].mean()))
    res = dict(acc=float(np.mean(correct)), p_correct=float(np.mean(pcorr)))
    if P > 1:
        by_pos = [float(pos_ok[pos_valid[:, j], j].mean()) for j in range(P)]
        res["acc_mean"] = float(pos_ok[pos_valid].mean())
        res["acc_by_position"] = by_pos
    if "supports" in cell.extra:
        pred = p[:, 0].argmax(-1)
        res["pred_in_input"] = float(np.mean(
            [pred[i] in cell.extra["supports"][i] for i in range(N)]))
    if "depth" in cell.extra:
        pred = p[:, 0].argmax(-1)
        tg = np.array([t[0] for t in cell.targets])
        dep = np.array(cell.extra["depth"])
        res["acc_by_burial_depth"] = {
            str(d): [float((pred[dep == d] == tg[dep == d]).mean()), int((dep == d).sum())]
            for d in sorted(set(dep.tolist()))}
    for k in ("query_letters_seen",):
        if k in cell.extra:
            res[k] = cell.extra[k]
    return res


def evaluate(paths, suite: str, device: torch.device, max_tokens: int | None = None,
             max_len: int | None = None, bf16: bool = False, keep_probs: bool = False,
             verbose: bool = True):
    """Score checkpoints ``paths`` (one per seed) on ``suite``.

    Returns ``(results, meta, probs)``: ``results[section][tag]`` holds per-seed
    metrics and, with several seeds, the probability-ensemble metrics.
    """
    max_tokens = max_tokens or (262144 if device.type == "cuda" else 32768)
    for p in paths:
        if not Path(p).is_file():
            raise SystemExit(f"checkpoint not found: {p}")
    names = [f"{Path(p).parent.name}/{Path(p).name}" for p in paths]
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False       # match the fp32 paper scoring
    models = [load_learner(p, device) for p in paths]
    if verbose:
        print(f"{len(models)} checkpoint(s) on {device}: {names}", flush=True)

    cells = [c for g in T.resolve(suite) for c in T.SUITES[g]()]
    if max_len:
        n_all = len(cells)
        cells = [c for c in cells if c.max_len <= max_len]
        print(f"--max-len {max_len}: scoring {len(cells)} cells, skipping {n_all - len(cells)}")

    results, probs_out = {}, {}
    t0 = time.time()
    for n, cell in enumerate(cells):
        ps = np.stack([cell_probs(m, cell, device, max_tokens, bf16) for m in models])
        entry = {"n_trials": len(cell.targets), "chance": cell.chance,
                 "per_seed": {nm: metrics(cell, p) for nm, p in zip(names, ps)}}
        if len(models) > 1:
            entry["ensemble"] = metrics(cell, ps.mean(0))
        results.setdefault(cell.section, {})[cell.tag] = entry
        if keep_probs:
            probs_out[cell.cid] = ps.astype(np.float16)
        if verbose:
            accs = [v["acc"] for v in entry["per_seed"].values()]
            ens = entry.get("ensemble", {}).get("acc_mean", entry.get("ensemble", {}).get("acc"))
            print(f"[{n + 1}/{len(cells)} {time.time() - t0:6.0f}s] {cell.cid:34s} "
                  f"seed acc {np.mean(accs):.3f} (min {min(accs):.3f} max {max(accs):.3f})"
                  + (f"  ens {ens:.3f}" if ens is not None else ""), flush=True)
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    meta = {"checkpoints": [str(p) for p in paths], "seeds": names, "suite": suite,
            "device": str(device), "bf16": bf16, "max_len": max_len,
            "seconds": round(time.time() - t0, 1),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return results, meta, probs_out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="checkpoint paths or globs; each is one seed")
    ap.add_argument("--suite", default="fig4",
                    help="raw | fig4 | printable | all | raw.sweep,raw.v4,...")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="tokens per forward (default 262144 on GPU, 32768 on CPU)")
    ap.add_argument("--max-len", type=int, default=None,
                    help="skip cells whose longest prompt exceeds this")
    ap.add_argument("--bf16", action="store_true",
                    help="bf16 autocast (the paper scored in fp32)")
    ap.add_argument("--probs", action="store_true", help="also save per-seed probs")
    ap.add_argument("--label", default="", help="free-text label stored in meta")
    args = ap.parse_args()

    paths = sorted({p for g in args.ckpts for p in (glob.glob(g) or [g])})
    results, meta, probs_out = evaluate(paths, args.suite, torch.device(args.device),
                                        args.max_tokens, args.max_len, args.bf16,
                                        args.probs)
    meta["label"] = args.label
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    if args.probs:
        np.savez_compressed(args.out.with_suffix(".probs.npz"),
                            **{k.replace("/", "__"): v for k, v in probs_out.items()})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
