"""WSD training with cooldown branches from one trunk, trunk checkpoints, sliding evaluation.

Protocol (ADR 0005 §3–5): the trunk runs at constant LR after warmup; for each branch end
E in (T, 2T, 4T, ...) a deep copy of model, optimizer and data stream cools down linearly to
zero over the last `cooldown_frac * E` steps. The trunk never sees the cooldowns: they use
their own copy of the data-stream RNG, so the trunk continues exactly as if they had not run.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .model import QLinear, TinyGPT


@dataclass(frozen=True)
class TrainSpec:
    tokens: int  # T: the first cooldown ends at T; later ones at 2T, 4T, ...
    branches: int = 3
    batch: int = 32
    lr: float = 3e-3
    wd: float = 0.1
    warmup_frac: float = 0.02  # of T (roadmap S2)
    cooldown_frac: float = 0.1  # of each branch's total steps (roadmap S2)
    seed: int = 0


def lr_at(step: int, warmup: int, peak: float, cd_start: int | None, cd_len: int) -> float:
    """Warmup-stable-decay: linear warmup, flat, linear to 0 over the cooldown."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if cd_start is None or step < cd_start:
        return peak
    return peak * max(0.0, 1 - (step - cd_start + 1) / cd_len)


class DataStream:
    """Random ctx+1 windows from a byte array; its RNG state is the position in the stream."""

    def __init__(self, data: np.ndarray, batch: int, ctx: int, seed: int) -> None:
        if len(data) < ctx + 2:
            raise ValueError("training data shorter than one window")
        self.data, self.batch, self.ctx = data, batch, ctx
        self.rng = np.random.default_rng(seed)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = self.rng.integers(0, len(self.data) - self.ctx - 1, size=self.batch)
        c = np.stack([self.data[i : i + self.ctx + 1] for i in ix]).astype(np.int64)
        c = torch.from_numpy(c)
        return c[:, :-1], c[:, 1:]

    def fork(self) -> DataStream:
        other = copy.copy(self)
        other.rng = copy.deepcopy(self.rng)
        return other

    def state(self) -> dict:
        return self.rng.bit_generator.state

    def set_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state


def eval_windows(n: int, ctx: int, stride: int) -> list[tuple[int, int, int]]:
    """(start, length, first scored offset) so that every target index 1..n-1 is scored once.

    A window starting at j of length L predicts targets j+1 .. j+L-1. Full windows advance by
    `stride`; one last window is aligned to the end to cover the tail; data shorter than a
    window is a single window.
    """
    if n < 2:
        return []
    if n <= ctx + 1:
        return [(0, n, 0)]
    out, done = [], 0  # done = last scored target index
    j = 0
    while j + ctx + 1 <= n:
        out.append((j, ctx + 1, done - j))
        done = j + ctx
        j += stride
    if done < n - 1:
        j = n - ctx - 1
        out.append((j, ctx + 1, done - j))
    return out


@torch.no_grad()
def sliding_bpb(
    model: torch.nn.Module,
    data: np.ndarray,
    n_bytes: int,
    stride: int | None = None,
    batch: int = 32,
) -> tuple[float, int]:
    """Bits per byte over the targets of data[:n_bytes], each scored once (roadmap S7).

    Returns (bpb, number of scored target bytes). Every scored target after the first window has
    at least ctx - stride bytes of context.
    """
    ctx = model.cfg.ctx
    stride = stride or ctx // 2
    arr = torch.from_numpy(np.asarray(data[: min(n_bytes, len(data))]).astype(np.int64))
    wins = eval_windows(arr.numel(), ctx, stride)
    if not wins:
        raise ValueError("need at least 2 bytes to evaluate")
    was = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(wins), batch):
        group = wins[i : i + batch]
        length = group[0][1]
        c = torch.stack([arr[j : j + length] for j, _, _ in group])
        nll = F.cross_entropy(model(c[:, :-1]).transpose(1, 2), c[:, 1:], reduction="none")
        for r, (_, _, first) in enumerate(group):
            total += nll[r, first:].sum().item()
            count += nll.size(1) - first
    model.train(was)
    return total / count / np.log(2), count


def _optimizer(model: TinyGPT, spec: TrainSpec) -> torch.optim.Optimizer:
    q = [m.weight for m in model.modules() if isinstance(m, QLinear)]
    ids = {id(p) for p in q}
    other = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW(
        [{"params": q, "weight_decay": spec.wd}, {"params": other, "weight_decay": 0.0}],
        lr=spec.lr,
        betas=(0.9, 0.95),
    )


@dataclass
class Trunk:
    """Everything needed to continue the trunk deterministically."""

    model: TinyGPT
    opt: torch.optim.Optimizer
    stream: DataStream
    step: int = 0

    def checkpoint(self, spec: TrainSpec) -> dict:
        return {
            "model": copy.deepcopy(self.model.state_dict()),
            "opt": copy.deepcopy(self.opt.state_dict()),
            "step": self.step,
            "data_rng": copy.deepcopy(self.stream.state()),
            "torch_rng": torch.get_rng_state(),
            "spec": asdict(spec),
        }

    @classmethod
    def restore(cls, model: TinyGPT, data: np.ndarray, spec: TrainSpec, ckpt: dict) -> Trunk:
        if ckpt["spec"] != asdict(spec):
            raise ValueError("checkpoint was made with a different TrainSpec")
        model.load_state_dict(ckpt["model"])
        opt = _optimizer(model, spec)
        opt.load_state_dict(ckpt["opt"])
        stream = DataStream(data, spec.batch, model.cfg.ctx, spec.seed)
        stream.set_state(ckpt["data_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        return cls(model, opt, stream, ckpt["step"])


def _step(model: TinyGPT, opt: torch.optim.Optimizer, stream: DataStream, lr: float) -> float:
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = stream.next()
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def schedule(spec: TrainSpec, ctx: int) -> dict:
    """Step counts of the protocol, with tokens = steps * batch * ctx exactly."""
    per_step = spec.batch * ctx
    T = max(1, spec.tokens // per_step)
    ends = [T * 2**k for k in range(spec.branches)]
    cds = [max(1, int(spec.cooldown_frac * e)) for e in ends]
    return {
        "per_step": per_step,
        "T": T,
        "warmup": max(1, int(spec.warmup_frac * T)),
        "ends": ends,
        "cooldowns": cds,
        "trunk_steps": ends[-1] - cds[-1],
    }


def train_wsd(
    model: TinyGPT,
    data: np.ndarray,
    spec: TrainSpec,
    on_branch: Callable[[int, TinyGPT, dict], None],
    on_checkpoint: Callable[[dict], None] | None = None,
    resume: dict | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run the trunk; at each branch point fork, cool down, report, discard the fork.

    Returns compute accounting: trunk and per-cooldown steps/tokens and wall seconds.
    """
    sch = schedule(spec, model.cfg.ctx)
    if resume is None:
        trunk = Trunk(
            model, _optimizer(model, spec), DataStream(data, spec.batch, model.cfg.ctx, spec.seed)
        )
    else:
        trunk = Trunk.restore(model, data, spec, resume)
    acct = {"trunk_steps": 0, "cooldowns": [], "trunk_seconds": 0.0}
    for end, cd_len in zip(sch["ends"], sch["cooldowns"], strict=True):
        cd_start = end - cd_len
        if trunk.step > cd_start:
            continue  # resumed past this branch point
        t0 = time.time()
        while trunk.step < cd_start:
            _step(
                trunk.model,
                trunk.opt,
                trunk.stream,
                lr_at(trunk.step, sch["warmup"], spec.lr, None, 0),
            )
            trunk.step += 1
            acct["trunk_steps"] += 1
        acct["trunk_seconds"] += time.time() - t0
        ckpt = trunk.checkpoint(spec)
        if on_checkpoint is not None:
            on_checkpoint(ckpt)
        branch = copy.deepcopy(trunk.model)
        bopt = _optimizer(branch, spec)
        bopt.load_state_dict(copy.deepcopy(trunk.opt.state_dict()))
        bstream = trunk.stream.fork()
        t0 = time.time()
        for s in range(cd_start, end):
            _step(branch, bopt, bstream, lr_at(s, sch["warmup"], spec.lr, cd_start, cd_len))
        info = {
            "end_step": end,
            "cooldown_steps": cd_len,
            "cooldown_start": cd_start,
            "tokens_seen": end * sch["per_step"],
            "cooldown_seconds": time.time() - t0,
        }
        acct["cooldowns"].append(info)
        log(f"branch end={end} tokens={end * sch['per_step']:,}")
        on_branch(end, branch, info)
        del branch, bopt, bstream
    acct["trunk_tokens"] = acct["trunk_steps"] * sch["per_step"]
    return acct
