#!/usr/bin/env python3
"""In-loop direction from matched pairs, teacher-forced alpha check, edited checkpoints (Qwen3-1.7B, thinking on).

Inputs: pairs.pt from loop_pairs_build.py (same prompt; one looping draw, one finished non-looping draw).
Blocks: 15, 16, 17, 18 (0-based decoder layers), where the per-block loop-vs-productive AUC peaked.

direction  For each build pair, run prompt + draw through the unedited model and read the residual stream after each
           block. Loop side: mean over the loop tokens of the looping draw (positions covered by a 30-gram occurring
           >= 20 times; the first 4,096 such positions). Non-loop side: mean over every thinking token of the
           non-looping draw. Per pair take loop mean - non-loop mean, average over pairs, normalize: u_l.
           --loop-side last uses the last 4,096 loop positions instead (--tag names the outputs).
           Also saved: the same with the non-loop side restricted to its last 2,048 thinking tokens (diagnostic).
short      Astra's direction: their four constructed short pairs (same math problem and opening, then a productive
           continuation vs one doubt sentence repeated; texts from their fwd_4 request, already in Qwen chat format).
           Per pair: mean over the last 36 tokens of the repetitive text minus the same for the productive text,
           averaged over pairs, normalized. Saved as directions_short.pt.
Two ways to apply a direction (the arms):
  write    Rank-one removal on the block's MLP output weight (Astra's form):
           W_down,l <- W_down,l - alpha * u_l (u_l^T W_down,l), for l in 15..18, computed in fp32, stored in bf16.
  resid    Projection on the residual stream at run time: h_l <- h_l - alpha * u_l (u_l^T h_l) on the output of
           blocks 15..18 (alpha 1 removes the component). Generation uses lp_resid_collect.py with a spec file.
Control direction: a seeded Gaussian unit vector per block, same alpha.
tf         On check pairs, log-probability of each actual next token, unedited and edited (--arm write|resid):
           loop tokens of the looping draw (first 2,048 loop positions) and thinking tokens of the non-looping draw
           (first 16,384). Reports per-pair mean change, then mean and quartiles over pairs. Also Astra's transfer
           probe (6 new short pairs): change on the tokens after the branch point of the repetitive and productive text.
export     Writes edited checkpoints (safetensors + tokenizer) for the requested (direction, alpha) conditions.
resid-spec Writes the spec files (blocks, u, alpha) the residual arm loads at generation time.

  loop_pairs_edit.py direction --pairs P --out D
  loop_pairs_edit.py tf --out D --alphas 0.5,1,2,4,8
  loop_pairs_edit.py export --out D --cond learned:2 --cond random:2
  loop_pairs_edit.py short --out D --astra-dir A
  loop_pairs_edit.py resid-spec --out D --tag _short --cond learned:1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = None
BLOCKS = (15, 16, 17, 18)
LOOP_POS_DIR = 4096
LOOP_POS_TF = 2048
PROD_TF = 16384
LATE = 2048
SHORT_LAST = 36
SEED = 20261104


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def configure(model_id: str | None = None, blocks: str | None = None, revision: str | None = None):
    """Point the module at another model (HF id or local path) and/or edit blocks ("15,19"); defaults stay Qwen3-1.7B, 15-18."""
    global MODEL_ID, MODEL_REVISION, BLOCKS
    if model_id:
        MODEL_ID = model_id
    if revision:
        MODEL_REVISION = revision
    if blocks:
        BLOCKS = tuple(int(x) for x in str(blocks).split(","))


def load_model():
    """HF model in bf16. Qwen3.5 checkpoints are image-text-to-text models (AutoModelForCausalLM fails on them); their
    decoder is model.model.language_model. Use decoder_layers() / hidden_size() rather than model.model.layers."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    path = MODEL_ID if os.path.isdir(MODEL_ID) else snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    arch = (AutoConfig.from_pretrained(path).architectures or [""])[0]
    cls = AutoModelForImageTextToText if arch.endswith("ForConditionalGeneration") else AutoModelForCausalLM
    m = cls.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    m.requires_grad_(False)
    return m, path


def decoder_layers(model):
    inner = model.model
    return inner.language_model.layers if hasattr(inner, "language_model") else inner.layers


def hidden_size(model) -> int:
    return model.config.get_text_config().hidden_size


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def astra_texts(astra_dir, tok, name):
    """Astra pair texts tokenized as written (chat tokens are in the text). Returns [(ids, char offsets)]."""
    texts = json.loads((Path(astra_dir) / name).read_text())["texts"]
    out = []
    for t in texts:
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        out.append((t, enc["input_ids"], [o[0] for o in enc["offset_mapping"]]))
    return out


def branch_start(a: str, b: str, offs: list[int]) -> int:
    """First token of a text that starts at or after the first character where a and b differ."""
    c = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    return next(i for i, o in enumerate(offs) if o >= c)


class _Stop(Exception):
    pass


class Capture:
    def __init__(self, model, stop_after: int | None):
        self.h = {}
        self.handles = []
        for l in BLOCKS:
            self.handles.append(decoder_layers(model)[l].register_forward_hook(self._hook(l, stop_after)))

    def _hook(self, l, stop_after):
        def f(_m, _i, o):
            self.h[l] = (o[0] if isinstance(o, tuple) else o)[0].float()
            if stop_after is not None and l == stop_after:
                raise _Stop

        return f

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def residuals(model, ids: list[int], stop: bool = True):
    cap = Capture(model, BLOCKS[-1] if stop else None)
    try:
        model(torch.tensor([ids], device="cuda"), use_cache=False)
    except _Stop:
        pass
    cap.close()
    return cap.h


def cmd_direction(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [p for p in torch.load(a.pairs) if p["split"] == "build"]
    model, _ = load_model()
    sums = {k: {l: torch.zeros(2048, dtype=torch.float64) for l in BLOCKS} for k in ("all", "late")}
    per_ds = {}
    n = 0
    for i, p in enumerate(pairs):
        P = len(p["prompt"])
        lpos = p["loop_pos"][-LOOP_POS_DIR:] if a.loop_side == "last" else p["loop_pos"][:LOOP_POS_DIR]
        hl = residuals(model, p["prompt"] + p["loop_toks"][: lpos[-1] + 1])
        hp = residuals(model, p["prompt"] + p["prod_toks"][: p["prod_think_end"]])
        idx = torch.tensor(lpos, device="cuda") + P
        te = p["prod_think_end"]
        for l in BLOCKS:
            ml = hl[l][idx].mean(0)
            mp_all = hp[l][P : P + te].mean(0)
            mp_late = hp[l][P + max(0, te - LATE) : P + te].mean(0)
            sums["all"][l] += (ml - mp_all).double().cpu()
            sums["late"][l] += (ml - mp_late).double().cpu()
            per_ds.setdefault(p["dataset"], {l: torch.zeros(2048, dtype=torch.float64) for l in BLOCKS})[l] += (ml - mp_all).double().cpu()
        n += 1
        if i % 20 == 0:
            log("direction", i, len(pairs), p["dataset"], p["sample_id"], "loop_len", lpos[-1] + 1, "prod_think", te)
    D = {k: {l: (v / n).float() for l, v in s.items()} for k, s in sums.items()}
    U = {l: D["all"][l] / D["all"][l].norm() for l in BLOCKS}
    R = random_dirs()
    diag = {
        "n_build_pairs": n,
        "per_block": {
            l: {
                "diff_norm": float(D["all"][l].norm()),
                "cos_all_vs_late": float(torch.nn.functional.cosine_similarity(D["all"][l], D["late"][l], dim=0)),
                "cos_math_vs_lcb": float(torch.nn.functional.cosine_similarity(per_ds["math_level5"][l].float(), per_ds["livecodebench"][l].float(), dim=0)),
                "cos_random_vs_learned": float(R[l] @ U[l]),
            }
            for l in BLOCKS
        },
        "cos_between_blocks": {f"{a}-{b}": float(U[a] @ U[b]) for a in BLOCKS for b in BLOCKS if a < b},
    }
    torch.save({"learned": U, "random": R, "diff_all": D["all"], "diff_late": D["late"]}, out / f"directions{a.tag}.pt")
    (out / f"direction_diag{a.tag}.json").write_text(json.dumps(diag, indent=1))
    log(json.dumps(diag))


def random_dirs(dim: int = 2048):
    g = torch.Generator().manual_seed(SEED)
    R = {}
    for l in BLOCKS:
        r = torch.randn(dim, generator=g)
        R[l] = r / r.norm()
    return R


def cmd_short(a):
    out = Path(a.out)
    model, path = load_model()
    tok = load_tokenizer(path)
    T = astra_texts(a.astra_dir, tok, "fwd4_request.json")
    last = {l: torch.zeros(2048, dtype=torch.float64) for l in BLOCKS}
    branch = {l: torch.zeros(2048, dtype=torch.float64) for l in BLOCKS}
    info = []
    for i in range(0, len(T), 2):
        (tp, ip, op), (tr, ir, orr) = T[i], T[i + 1]
        hp, hr = residuals(model, ip), residuals(model, ir)
        bp, br = branch_start(tp, tr, op), branch_start(tr, tp, orr)
        info.append({"pair": i // 2, "prod_tokens": len(ip), "rep_tokens": len(ir), "prod_branch_tokens": len(ip) - bp, "rep_branch_tokens": len(ir) - br})
        for l in BLOCKS:
            last[l] += (hr[l][-SHORT_LAST:].mean(0) - hp[l][-SHORT_LAST:].mean(0)).double().cpu()
            branch[l] += (hr[l][br:].mean(0) - hp[l][bp:].mean(0)).double().cpu()
    U = {l: (last[l] / last[l].norm()).float() for l in BLOCKS}
    first, lastk = torch.load(out / "directions.pt")["learned"], torch.load(out / "directions_last.pt")["learned"]
    cos = lambda x, y: float(x.float() @ y.float() / (x.float().norm() * y.float().norm()))
    diag = {"pairs": info, "last_tokens": SHORT_LAST,
            "per_block": {l: {"diff_norm": float(last[l].norm() / len(info)), "cos_last36_vs_branch_span": cos(last[l], branch[l]),
                              "cos_vs_first4096": cos(U[l], first[l]), "cos_vs_last4096": cos(U[l], lastk[l])} for l in BLOCKS},
            "cos_between_blocks": {f"{x}-{y}": float(U[x] @ U[y]) for x in BLOCKS for y in BLOCKS if x < y}}
    torch.save({"learned": U, "random": random_dirs(), "diff_last36": last, "diff_branch": branch}, out / "directions_short.pt")
    (out / "direction_diag_short.json").write_text(json.dumps(diag, indent=1))
    log(json.dumps(diag))


class Resid:
    """Run-time arm: h_l <- h_l - alpha * u_l (u_l^T h_l) on the output of each block in BLOCKS."""

    def __init__(self, model, U, alpha):
        self.handles = [model.model.layers[l].register_forward_hook(self._hook(U[l].to("cuda", torch.float32), alpha)) for l in BLOCKS]

    @staticmethod
    def _hook(u, alpha):
        def f(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.float()
            h2 = (hf - alpha * (hf @ u)[..., None] * u).to(h.dtype)
            return (h2, *o[1:]) if isinstance(o, tuple) else h2

        return f

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def apply_edit(model, U, alpha, saved=None):
    for l in BLOCKS:
        W = model.model.layers[l].mlp.down_proj.weight
        if saved is not None:
            saved[l] = W.detach().clone()
        u = U[l].to(W.device, torch.float32)
        Wf = W.float()
        W.copy_((Wf - alpha * torch.outer(u, u @ Wf)).to(W.dtype))


@torch.no_grad()
def restore(model, saved):
    for l, w in saved.items():
        model.model.layers[l].mlp.down_proj.weight.copy_(w)


@torch.no_grad()
def token_logprobs(model, ids: list[int], positions: torch.Tensor) -> torch.Tensor:
    """log p(ids[t] | ids[:t]) at the given absolute positions t."""
    out = model.model(torch.tensor([ids], device="cuda"), use_cache=False).last_hidden_state[0]
    tgt = torch.tensor(ids, device="cuda")[positions]
    res = []
    for s in range(0, len(positions), 4096):
        pos = positions[s : s + 4096]
        logits = model.lm_head(out[pos - 1]).float()
        res.append(logits.log_softmax(-1).gather(1, tgt[s : s + 4096, None])[:, 0])
    return torch.cat(res)


def cmd_tf(a):
    out = Path(a.out)
    dirs = torch.load(out / f"directions{a.tag}.pt")
    pairs = [p for p in torch.load(a.pairs) if p["split"] == "check"]
    model, path = load_model()
    tok = load_tokenizer(path)
    probe = []
    T = astra_texts(a.astra_dir, tok, "transfer_probe_request.json")[:12]
    for i in range(0, 12, 2):
        (tp, ip, op), (tr, ir, orr) = T[i], T[i + 1]
        bp, br = branch_start(tp, tr, op), branch_start(tr, tp, orr)
        probe.append((ir, torch.arange(br, len(ir), device="cuda"), ip, torch.arange(bp, len(ip), device="cuda")))
    items = []
    for p in pairs:
        P = len(p["prompt"])
        lpos = p["loop_pos"][:LOOP_POS_TF]
        te = min(p["prod_think_end"], PROD_TF)
        items.append(dict(ds=p["dataset"], sid=p["sample_id"],
                          loop_ids=p["prompt"] + p["loop_toks"][: lpos[-1] + 1],
                          loop_pos=torch.tensor(lpos, device="cuda") + P,
                          prod_ids=p["prompt"] + p["prod_toks"][:te],
                          prod_pos=torch.arange(P, P + te, device="cuda")))

    def run():
        r = []
        for it in items:
            r.append((float(token_logprobs(model, it["loop_ids"], it["loop_pos"]).mean()),
                      float(token_logprobs(model, it["prod_ids"], it["prod_pos"]).mean())))
        return r

    def run_probe():
        return [(float(token_logprobs(model, ir, pr).mean()), float(token_logprobs(model, ip, pp).mean())) for ir, pr, ip, pp in probe]

    base, pbase = run(), run_probe()
    log("base done", len(items))
    alphas = [float(x) for x in a.alphas.split(",")]
    results = {"arm": a.arm, "n_check_pairs": len(items), "base": {"loop_mean": sum(b[0] for b in base) / len(base), "prod_mean": sum(b[1] for b in base) / len(base),
               "astra_probe_rep_mean": sum(b[0] for b in pbase) / len(pbase), "astra_probe_prod_mean": sum(b[1] for b in pbase) / len(pbase)}, "conditions": []}
    for direction in ("learned", "random"):
        for alpha in alphas:
            if a.arm == "write":
                saved = {}
                apply_edit(model, dirs[direction], alpha, saved)
                r, pr = run(), run_probe()
                restore(model, saved)
            else:
                hk = Resid(model, dirs[direction], alpha)
                r, pr = run(), run_probe()
                hk.close()
            dl = sorted(x[0] - b[0] for x, b in zip(r, base))
            dp = sorted(x[1] - b[1] for x, b in zip(r, base))
            q = lambda v: [v[len(v) // 4], v[len(v) // 2], v[3 * len(v) // 4]]
            by_ds = {}
            for ds in ("math_level5", "livecodebench"):
                ii = [i for i, it in enumerate(items) if it["ds"] == ds]
                by_ds[ds] = {"d_loop": sum(r[i][0] - base[i][0] for i in ii) / len(ii), "d_prod": sum(r[i][1] - base[i][1] for i in ii) / len(ii), "n": len(ii)}
            row = {"direction": direction, "alpha": alpha, "d_loop_mean": sum(dl) / len(dl), "d_loop_q": q(dl),
                   "d_prod_mean": sum(dp) / len(dp), "d_prod_q": q(dp), "by_dataset": by_ds,
                   "astra_probe_d_rep": sum(x[0] - b[0] for x, b in zip(pr, pbase)) / len(pr),
                   "astra_probe_d_prod": sum(x[1] - b[1] for x, b in zip(pr, pbase)) / len(pr),
                   "per_pair": [{"ds": it["ds"], "sid": it["sid"], "d_loop": x[0] - b[0], "d_prod": x[1] - b[1]} for it, x, b in zip(items, r, base)]}
            results["conditions"].append(row)
            log(a.arm, direction, alpha, "d_loop", round(row["d_loop_mean"], 4), "d_prod", round(row["d_prod_mean"], 4),
                "probe d_rep", round(row["astra_probe_d_rep"], 4), "d_prod", round(row["astra_probe_d_prod"], 4))
            (out / f"tf_alpha{a.tag}_{a.arm}.json").write_text(json.dumps(results, indent=1))


def cmd_export(a):
    out = Path(a.out)
    dirs = torch.load(out / f"directions{a.tag}.pt")
    model, path = load_model()
    for cond in a.cond:
        direction, alpha = cond.split(":")
        dest = out / "checkpoints" / f"{direction}{a.tag}_a{alpha}"
        saved = {}
        apply_edit(model, dirs[direction], float(alpha), saved)
        model.save_pretrained(dest, safe_serialization=True)
        restore(model, saved)
        for f in os.listdir(path):
            if f.startswith("tokenizer") or f in ("vocab.json", "merges.txt", "generation_config.json"):
                shutil.copy(os.path.join(path, f), dest / f)
        (dest / "EDIT.json").write_text(json.dumps({"base": MODEL_ID, "blocks": BLOCKS, "direction": direction, "alpha": float(alpha),
                                                    "formula": "W_down - alpha * outer(u, u @ W_down)"}))
        log("exported", dest)


def cmd_resid_spec(a):
    out = Path(a.out)
    dirs = torch.load(out / f"directions{a.tag}.pt")
    (out / "resid").mkdir(exist_ok=True)
    for cond in a.cond:
        direction, alpha = cond.split(":")
        dest = out / "resid" / f"{direction}{a.tag}_a{alpha}.pt"
        torch.save({"blocks": list(BLOCKS), "u": {l: dirs[direction][l].float() for l in BLOCKS}, "alpha": float(alpha),
                    "formula": "h_l - alpha * u_l (u_l . h_l) at block output"}, dest)
        log("resid spec", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["direction", "short", "tf", "export", "resid-spec"])
    ap.add_argument("--pairs", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", default="0.5,1,2,4,8")
    ap.add_argument("--cond", action="append", default=[])
    ap.add_argument("--loop-side", choices=["first", "last"], default="first", help="first or last LOOP_POS_DIR loop-token positions")
    ap.add_argument("--tag", default="", help="suffix for directions/tf/checkpoint names, e.g. _last, _short")
    ap.add_argument("--arm", choices=["write", "resid"], default="write")
    ap.add_argument("--astra-dir", default="", help="dir with Astra's fwd4_request.json and transfer_probe_request.json")
    a = ap.parse_args()
    {"direction": cmd_direction, "short": cmd_short, "tf": cmd_tf, "export": cmd_export, "resid-spec": cmd_resid_spec}[a.cmd](a)


if __name__ == "__main__":
    main()
