"""Shared helpers: model construction, checkpoint I/O, and a lean forward pass.

Every arm (self-play and uniform-prior checkpoints from Hugging Face, and our
own DCLM-trained learners) uses the paper's ``ProgramLanguageModel`` and the
same on-disk layout, so the ICL harness and the scorer treat them identically:

    <root>/<arm>/<size>/seed-<seed>/config.json          {"learner": {...}}
    <root>/<arm>/<size>/seed-<seed>/learner_<tag>.pth    {"learner_state_dict", ...}
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SCORING = PROJECT / "self_play_pretraining" / "scoring"
sys.path.insert(0, str(SCORING))
from src.framework.model import ProgramLanguageModel  # noqa: E402

HF_REPO = "nourya-cohen/solomonoff-paper"
CONTEXT = 4096
PREFIX = ord("O")

# Ladder rungs (paper Table / HF model card). N_NONEMB is the compute-axis N.
ARCH = {
    "100k": dict(d_model=64, n_heads=1, n_layers=1),
    "500k": dict(d_model=128, n_heads=2, n_layers=2),
    "1M": dict(d_model=128, n_heads=2, n_layers=4),
    "3M": dict(d_model=256, n_heads=4, n_layers=4),
    "6M": dict(d_model=256, n_heads=4, n_layers=8),
    "24M": dict(d_model=512, n_heads=8, n_layers=8),
}
COMMON = dict(max_len=CONTEXT, vocab_size=256, base_d_model=16)
N_NONEMB = {"100k": 65_728, "500k": 492_160, "1M": 984_192,
            "3M": 3_016_960, "6M": 6_033_664, "24M": 24_253_184}
# Self-play learner tokens per round (figures/fig2 README: 1536 programs x 4096).
SP_TOKENS_PER_ROUND = 1536 * 4096


def sp_tokens(round_: int) -> int:
    """Learner tokens consumed by self-play after checkpoint ``round_``."""
    return SP_TOKENS_PER_ROUND * (round_ + 1)


# Self-play checkpoints the comparison uses (the released ones are 0, 256, ...,
# 8191; 2816 is the paper's ICL point) and the DCLM token counts matching them.
SP_ROUNDS = (0, 256, 512, 1024, 2048, 2816, 8191)
MATCH_TOKENS = tuple(sp_tokens(r) for r in (256, 512, 1024, 2048, 2816))  # 1.6B ... 17.7B


def load_env(path: Path = PROJECT / ".env") -> None:
    """Load KEY=VALUE lines from the project .env (existing variables win)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def wandb_init(enabled: bool, resume_file: Path | None = None, **kwargs):
    """``wandb.init`` if enabled and a key is available, else None.

    ``resume_file`` stores the run id so a restarted job continues the same
    W&B run instead of opening a new one.
    """
    if not enabled:
        return None
    load_env()
    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY not set; logging disabled")
        return None
    import wandb
    run_id = None
    if resume_file is not None and resume_file.is_file():
        run_id = resume_file.read_text().strip()
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "icl-selfplay"),
                     id=run_id, resume="allow" if run_id else None, **kwargs)
    if resume_file is not None:
        resume_file.write_text(run.id)
    return run


def arch_config(size: str) -> dict:
    return {**ARCH[size], **COMMON}


def hf_token() -> str | None:
    """HF token from the usual variables (Prime secret is HUGGINGFACE_API_KEY)."""
    for k in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    return None


def build_model(cfg: dict) -> ProgramLanguageModel:
    return ProgramLanguageModel(**cfg)


def read_config(ckpt: Path) -> dict:
    """The learner config next to a checkpoint (HF layout), else infer from dir."""
    cfg_path = Path(ckpt).parent / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        return cfg.get("learner", cfg)
    size = Path(ckpt).parent.parent.name
    if size in ARCH:
        return arch_config(size)
    raise FileNotFoundError(f"no config.json next to {ckpt} and size unknown")


def load_learner(ckpt: str | Path, device="cpu", dtype=torch.float32) -> ProgramLanguageModel:
    ckpt = Path(ckpt)
    model = build_model(read_config(ckpt))
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob.get("learner_state_dict", blob.get("model_state_dict", blob))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # inv_freq / f_token_mask are non-persistent buffers; anything else is a bug.
    bad = [k for k in missing if not k.endswith(("inv_freq", "f_token_mask"))]
    if bad or unexpected:
        raise RuntimeError(f"{ckpt}: missing={bad} unexpected={unexpected}")
    return model.eval().to(device=device, dtype=dtype)


def save_learner(model, path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = path.parent / "config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps({"learner": model.get_config()}, indent=1))
    sd = {k: v.detach().to("cpu", torch.float32) for k, v in model.state_dict().items()}
    tmp = path.with_suffix(".tmp")
    torch.save({**meta, "learner_state_dict": sd}, tmp)
    tmp.replace(path)


def hidden_states(model: ProgramLanguageModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Final-norm hidden states ``[B, T, d]`` without the program-row masking.

    For rows that start with the output prefix ``O`` (every row we train or
    evaluate on) ``ProgramLanguageModel.forward`` applies no mask, so
    ``lm_head(hidden_states(...))`` equals its logits exactly. Skipping the full
    ``[B, T, 256]`` masked logits lets callers project only the positions they
    need.
    """
    T = input_ids.shape[1]
    positions = torch.arange(T, device=input_ids.device)
    x = model.wte(input_ids)
    pos_emb = model.rotary_emb(x, positions)
    for block in model.blocks:
        x = block(x, pos_emb)
    return model.ln_f(x)


def init_weights(model: ProgramLanguageModel, std: float = 0.02) -> None:
    """GPT-2-style init (the paper's own init is unpublished).

    Normal(0, std) for embeddings and linears; residual output projections
    (attention ``o_proj`` and MLP ``down_proj``) scaled by 1/sqrt(2 * n_layers).
    """
    resid_std = std / math.sqrt(2 * model.n_layers)
    for name, p in model.named_parameters():
        if p.dim() < 2:
            continue                                   # RMSNorm weights stay 1
        s = resid_std if name.endswith(("o_proj.weight", "down_proj.weight")) else std
        torch.nn.init.normal_(p, mean=0.0, std=s)


def lm_loss(model, seq: torch.Tensor) -> torch.Tensor:
    """Mean next-byte cross-entropy (nats) of ``seq[:, 1:]`` given ``seq[:, :-1]``."""
    h = hidden_states(model, seq[:, :-1])
    logits = model.lm_head(h)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           seq[:, 1:].reshape(-1))
