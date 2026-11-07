#!/usr/bin/env python3
"""Experiment 1: learn the sensing vector s of a rank-one MLP-output edit, steering vector v fixed (Qwen3-1.7B).

Model and blocks: Qwen3-1.7B, blocks 15-18 by default; --model (HF id or local path), --revision and --blocks (e.g.
"15,19") point every command at another model, as for Qwen3.5-0.8B at blocks 15 and 19 (Astra's regime).
Edit at the chosen blocks (no bias), on the MLP output x of each block:  x <- x - v_l (s_l . x).
This is exactly the weight edit W_down <- W_down - outer(v_l, s_l @ W_down). The s = v baseline is s_l = alpha * v_l.

Pairs: scripts/steer/loop_short_pairs.json (Astra's 4 fwd_4 pairs first, then 28 more in the same format). N = 4 uses
Astra's pairs, N = 32 all of them. Validation: Astra's 6 transfer-probe pairs (first 12 texts of transfer_probe_request.json).

direction  v_l from the first N pairs: mean over the last 36 tokens of the residual stream after block l, repetitive text
           minus productive text, averaged over pairs, unit norm (as loop_pairs_edit.py short). Writes v_n{N}.pt.
train      Learn s (start 0, Adam lr 3e-3, 24 full-batch updates over the N pairs; the CLAS head recipe). Losses, each a
           per-text token mean, averaged over pairs:
             ntp  next-token loss on the productive text's assistant tokens (everything after the assistant header)
             ul   unlikelihood -log(1 - p(token)) on the repetitive text's tokens from the 2nd copy of the doubt sentence on
             kl   KL(unedited || edited) next-token distributions on the productive text's assistant tokens
           objective  clas = ntp + beta*kl;  ul = ul + beta*kl;  contrast = ntp + ul + beta*kl.
           The same objective is computed on the validation pairs at step 0 (s = 0) and after every update; the
           lowest-validation step is kept (step 0 included). Writes s/{obj}_n{N}_b{beta}.pt with the full history.
export     Edited checkpoints: --cond sv:N:alpha (s = alpha v) or --cond {obj}:N:beta (learned s at the kept step);
           --cond {obj}:N:beta@STEP takes s after update STEP instead (checkpoint {obj}_n{N}_b{beta}_s{STEP}).

  loop_s_learn.py direction --out D --astra-dir A --n 4
  loop_s_learn.py train --out D --astra-dir A --n 4 --obj clas --beta 0
  loop_s_learn.py export --out D --cond sv:4:2 --cond clas:4:0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_pairs_edit as LPE  # noqa: E402
from loop_pairs_edit import SHORT_LAST, decoder_layers, hidden_size, load_model, load_tokenizer, log, random_dirs, residuals  # noqa: E402

PAIRS = Path(__file__).resolve().parent / "loop_short_pairs.json"
HEADER = "<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<think>\n{opening} "
ASSIST = "<|im_start|>assistant\n"
LR, STEPS = 3e-3, 24


def pair_texts(p):
    h = HEADER.format(**p)
    return h + p["prod"], h + " ".join([p["unit"]] * p["n_rep"])


def load_pairs(astra_dir):
    ps = json.loads(PAIRS.read_text())["pairs"]
    astra = json.loads((Path(astra_dir) / "fwd4_request.json").read_text())["texts"]
    for i in range(4):  # Astra's pairs rebuilt verbatim
        assert pair_texts(ps[i]) == (astra[2 * i], astra[2 * i + 1]), i
    return [pair_texts(p) for p in ps]


def val_texts(astra_dir):
    t = json.loads((Path(astra_dir) / "transfer_probe_request.json").read_text())["texts"][:12]
    return [(t[i], t[i + 1]) for i in range(0, 12, 2)]


def second_copy_char(prod: str, rep: str) -> tuple[int, int, int]:
    """(branch start char, char where the 2nd copy of the repeated sentence starts, copies) for a pair."""
    c = next(i for i, (x, y) in enumerate(zip(prod, rep)) if x != y)
    s0 = rep.rfind(" ", 0, c) + 1
    tail = rep[s0:]
    for k in range(8, 1, -1):
        L = (len(tail) + 1) // k - 1
        if L > 0 and " ".join([tail[:L]] * k) == tail:
            return s0, s0 + L + 1, k
    raise ValueError(f"no repeated unit in: {tail!r}")


def encode(tok, text):
    e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return e["input_ids"], [o[0] for o in e["offset_mapping"]]


def build_items(tok, pairs):
    """Per pair: token ids and target positions (predicted tokens) for ntp/kl (prod) and ul (rep) and probe tails."""
    items = []
    for prod, rep in pairs:
        ip, op = encode(tok, prod)
        ir, orr = encode(tok, rep)
        s0, c2, k = second_copy_char(prod, rep)
        a = prod.index(ASSIST) + len(ASSIST)
        pos_ntp = [t for t in range(1, len(ip)) if op[t] >= a]
        pos_ul = [t for t in range(1, len(ir)) if orr[t] >= c2 - 1]  # the token holding the space before copy 2 on
        tail_r = [t for t in range(1, len(ir)) if orr[t] >= s0 - 1]
        tail_p = [t for t in range(1, len(ip)) if op[t] >= s0 - 1]
        items.append(dict(ip=ip, ir=ir, pos_ntp=pos_ntp, pos_ul=pos_ul, tail_r=tail_r, tail_p=tail_p, copies=k))
    return items


class Edit:
    """x_l <- x_l - v_l (s_l . x_l) on the MLP output of each block in BLOCKS; s_l are leaf tensors."""

    def __init__(self, model, V, S):
        self.handles = [decoder_layers(model)[l].mlp.register_forward_hook(self._hook(V[l], S[l])) for l in LPE.BLOCKS]

    @staticmethod
    def _hook(v, s):
        def f(_m, _i, o):
            xf = o.float()
            return (xf - (xf @ s)[..., None] * v).to(o.dtype)

        return f

    def close(self):
        for h in self.handles:
            h.remove()


def logprobs(model, ids, pos):
    out = model(torch.tensor([ids], device="cuda"), use_cache=False).logits[0]
    return out[torch.tensor(pos, device="cuda") - 1].float().log_softmax(-1)


def losses(model, it, base_lp, need):
    tgt = lambda ids, pos: torch.tensor([ids[t] for t in pos], device="cuda")[:, None]
    r = {}
    if need & {"ntp", "kl"}:
        lp = logprobs(model, it["ip"], it["pos_ntp"])
        r["ntp"] = -lp.gather(1, tgt(it["ip"], it["pos_ntp"])).mean()
        b = base_lp[0]
        r["kl"] = (b.exp() * (b - lp)).sum(-1).mean()
    if "ul" in need:
        lp = logprobs(model, it["ir"], it["pos_ul"])
        p = lp.gather(1, tgt(it["ir"], it["pos_ul"])).exp().clamp(max=1 - 1e-6)
        r["ul"] = -torch.log1p(-p).mean()
    return r


def objective(r, obj, beta):
    k = beta * r["kl"] if beta else 0.0
    if obj == "clas":
        return r["ntp"] + k
    if obj == "ul":
        return r["ul"] + k
    return r["ntp"] + r["ul"] + k


@torch.no_grad()
def probe(model, its):
    """Mean log-prob of the tokens after the branch point, repetitive and productive text."""
    rr, pp = [], []
    for it in its:
        for ids, pos, acc in ((it["ir"], it["tail_r"], rr), (it["ip"], it["tail_p"], pp)):
            lp = logprobs(model, ids, pos)
            acc.append(float(lp.gather(1, torch.tensor([ids[t] for t in pos], device="cuda")[:, None]).mean()))
    return sum(rr) / len(rr), sum(pp) / len(pp)


def cmd_direction(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, path = load_model()
    tok = load_tokenizer(path)
    pairs = load_pairs(a.astra_dir)[: a.n]
    B, D = LPE.BLOCKS, hidden_size(model)
    acc = {l: torch.zeros(D, dtype=torch.float64) for l in B}
    for prod, rep in pairs:
        hp, hr = residuals(model, encode(tok, prod)[0]), residuals(model, encode(tok, rep)[0])
        for l in B:
            acc[l] += (hr[l][-SHORT_LAST:].mean(0) - hp[l][-SHORT_LAST:].mean(0)).double().cpu()
    V = {l: (acc[l] / acc[l].norm()).float() for l in B}
    diag = {"model": LPE.MODEL_ID, "blocks": list(B), "n": a.n, "per_block": {l: {"diff_norm": float(acc[l].norm() / a.n)} for l in B}}
    ref = Path(a.ref_short) if a.ref_short else None
    if ref and ref.exists():
        S = torch.load(ref)["learned"]
        for l in B:
            diag["per_block"][l]["cos_vs_directions_short"] = float(V[l] @ S[l])
    torch.save({"learned": V, "random": random_dirs(D), "n": a.n, "model": LPE.MODEL_ID, "blocks": list(B)}, out / f"v_n{a.n}.pt")
    (out / f"v_n{a.n}_diag.json").write_text(json.dumps(diag, indent=1))
    log(json.dumps(diag))


def cmd_train(a):
    out = Path(a.out)
    (out / "s").mkdir(parents=True, exist_ok=True)
    name = f"{a.obj}_n{a.n}_b{a.beta:g}"
    V = {l: v.cuda() for l, v in torch.load(out / f"v_n{a.n}.pt")["learned"].items()}
    model, path = load_model()
    tok = load_tokenizer(path)
    tr = build_items(tok, load_pairs(a.astra_dir)[: a.n])
    va = build_items(tok, val_texts(a.astra_dir))
    need = {"clas": {"ntp", "kl"}, "ul": {"ul", "kl"} if a.beta else {"ul"}, "contrast": {"ntp", "ul", "kl"}}[a.obj]
    with torch.no_grad():  # unedited next-token log-probs on the productive assistant tokens, for the KL term
        btr = [(logprobs(model, it["ip"], it["pos_ntp"]),) for it in tr]
        bva = [(logprobs(model, it["ip"], it["pos_ntp"]),) for it in va]
        pbase = probe(model, va)
    BLOCKS = LPE.BLOCKS
    S = {l: torch.zeros(hidden_size(model), device="cuda", requires_grad=True) for l in BLOCKS}
    opt = torch.optim.Adam(list(S.values()), lr=LR)
    edit = Edit(model, V, S)
    model.gradient_checkpointing_disable()
    hist = []

    def evaluate(step, train_terms):
        with torch.no_grad():
            rs = [losses(model, it, b, need) for it, b in zip(va, bva)]
            terms = {k: float(sum(r[k] for r in rs) / len(rs)) for k in rs[0]}
            pr = probe(model, va)
        h = {"step": step, "val": float(objective(terms, a.obj, a.beta)) if all(k in terms for k in need) else None,
             "val_terms": terms, "train_terms": train_terms,
             "probe_d_rep": pr[0] - pbase[0], "probe_d_prod": pr[1] - pbase[1],
             "s_norm": {l: float(S[l].norm()) for l in BLOCKS},
             "cos_s_v": {l: float(S[l] @ V[l] / (S[l].norm() + 1e-12)) for l in BLOCKS},
             "s": {l: S[l].detach().cpu().clone() for l in BLOCKS}}
        hist.append(h)
        log(name, "step", step, "val", round(h["val"], 4), "train", {k: round(v, 4) for k, v in train_terms.items()},
            "probe d_rep", round(h["probe_d_rep"], 4), "d_prod", round(h["probe_d_prod"], 4),
            "|s|", [round(x, 3) for x in h["s_norm"].values()])

    evaluate(0, {})
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        tot = {}
        for it, b in zip(tr, btr):
            r = losses(model, it, b, need)
            (objective(r, a.obj, a.beta) / len(tr)).backward()
            for k, v in r.items():
                tot[k] = tot.get(k, 0.0) + float(v) / len(tr)
        opt.step()
        evaluate(step, tot)
    edit.close()
    best = min(hist, key=lambda h: (h["val"], h["step"]))
    torch.save({"obj": a.obj, "n": a.n, "beta": a.beta, "best_step": best["step"], "s": best["s"], "v": {l: v.cpu() for l, v in V.items()},
                "history": [{k: v for k, v in h.items() if k != "s"} for h in hist], "all_s": [h["s"] for h in hist]},
               out / "s" / f"{name}.pt")
    summ = {"name": name, "best_step": best["step"], "best_val": best["val"], "step0_val": hist[0]["val"],
            "best_probe_d_rep": best["probe_d_rep"], "best_probe_d_prod": best["probe_d_prod"],
            "best_s_norm": best["s_norm"], "best_cos_s_v": best["cos_s_v"],
            "history": [{k: v for k, v in h.items() if k != "s"} for h in hist]}
    (out / "s" / f"{name}.json").write_text(json.dumps(summ, indent=1))
    log(name, "best step", best["step"])


@torch.no_grad()
def cmd_export(a):
    out = Path(a.out)
    BLOCKS = LPE.BLOCKS
    model, path = load_model()
    if hasattr(model.model, "language_model"):  # Qwen3.5: patch the two tensors in a copy of the original files
        del model
        return export_patch(a, out, path)
    for cond in a.cond:
        V, S, name, meta, n = cond_vectors(out, cond)
        dest = out / "checkpoints" / name
        saved = {}
        for l in BLOCKS:
            W = model.model.layers[l].mlp.down_proj.weight
            saved[l] = W.detach().clone()
            v, s = V[l].to(W.device, torch.float32), S[l].to(W.device, torch.float32)
            W.copy_((W.float() - torch.outer(v, s @ W.float())).to(W.dtype))
        model.save_pretrained(dest, safe_serialization=True)
        for l, w in saved.items():
            model.model.layers[l].mlp.down_proj.weight.copy_(w)
        for f in os.listdir(path):
            if f.startswith("tokenizer") or f in ("vocab.json", "merges.txt", "generation_config.json"):
                shutil.copy(os.path.join(path, f), dest / f)
        (dest / "EDIT.json").write_text(json.dumps({"base": LPE.MODEL_ID, "blocks": BLOCKS, "cond": cond, "n_pairs": int(n), **meta,
                                                    "formula": "W_down - outer(v, s @ W_down)"}))
        log("exported", dest)


def cond_vectors(out, cond):
    kind, n, x = cond.split(":")
    V = torch.load(out / f"v_n{n}.pt")["learned"]
    if kind == "sv":
        return V, {l: float(x) * V[l] for l in LPE.BLOCKS}, f"sv_n{n}_a{x}", {"alpha": float(x)}, n
    x, _, step = x.partition("@")  # "ul:32:0.1@12": s after update 12 of training instead of the kept step
    d = torch.load(out / "s" / f"{kind}_n{n}_b{float(x):g}.pt")
    name = f"{kind}_n{n}_b{float(x):g}"
    if step:
        return V, d["all_s"][int(step)], f"{name}_s{int(step)}", {"beta": float(x), "step": int(step)}, n
    return V, d["s"], name, {"beta": float(x), "best_step": d["best_step"]}, n


def down_proj_name(l):
    return f"model.language_model.layers.{l}.mlp.down_proj.weight"


@torch.no_grad()
def export_patch(a, out, path):
    """Edited checkpoint = every file of the original checkpoint, with each edited block's down_proj replaced by
    bf16(W - outer(v, s @ W)) computed in fp32 (vLLM loads it unchanged)."""
    from safetensors.torch import load_file, save_file

    wmap = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]
    for cond in a.cond:
        V, S, name, meta, n = cond_vectors(out, cond)
        dest = out / "checkpoints" / name
        dest.mkdir(parents=True, exist_ok=True)
        shards = {wmap[down_proj_name(l)] for l in LPE.BLOCKS}
        rel = {}
        for f in os.listdir(path):
            src = Path(path) / f
            if src.is_file() and f not in shards:
                shutil.copy(src, dest / f)
        for shard in shards:
            sd = load_file(str(Path(path) / shard))
            for l in LPE.BLOCKS:
                k = down_proj_name(l)
                if k in sd:
                    W = sd[k].cuda().float()
                    v, s = V[l].cuda().float(), S[l].cuda().float()
                    E = (W - torch.outer(v, s @ W)).to(sd[k].dtype)
                    rel[k] = float((E.float() - W).norm() / W.norm())
                    sd[k] = E.cpu().contiguous()
            save_file(sd, str(dest / shard), metadata={"format": "pt"})
        (dest / "EDIT.json").write_text(json.dumps({"base": LPE.MODEL_ID, "revision": LPE.MODEL_REVISION, "blocks": list(LPE.BLOCKS),
                                                    "cond": cond, "n_pairs": int(n), **meta, "relative_change": rel,
                                                    "formula": "W_down - outer(v, s @ W_down)"}))
        log("exported", dest, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["direction", "train", "export"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--astra-dir", default="")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--obj", choices=["clas", "ul", "contrast"], default="clas")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--cond", action="append", default=[])
    ap.add_argument("--ref-short", default="", help="directions_short.pt from loop_pairs_edit.py, for a reproduction check at N=4")
    ap.add_argument("--model", default="", help="HF id or local path (default Qwen/Qwen3-1.7B)")
    ap.add_argument("--revision", default="")
    ap.add_argument("--blocks", default="", help="edited blocks, e.g. 15,19 (default 15,16,17,18)")
    a = ap.parse_args()
    LPE.configure(a.model or None, a.blocks or None, a.revision or None)
    {"direction": cmd_direction, "train": cmd_train, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
