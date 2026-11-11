#!/usr/bin/env python3
"""Natural-trace steering and sensing vectors for Qwen3-1.7B (blocks 15-18): the same recipe as loop_s_learn.py, with
the 32 short synthetic pairs replaced by 32 looping and 32 non-looping traces of the unedited model.

Traces: the unedited model's greedy dev rollouts (one per prompt, MATH level 5 and LiveCodeBench). Prompts whose
traces the stopping rule scores (loop_s_devloss.py's seeded draw from --stop-traces) are left out. With a fixed seed:
16 non-looping traces that close their thinking per dataset, and 32 looping traces (every looping MATH trace left, 9 in
our run; LiveCodeBench fills the rest). The prompts used here are then dropped from dev for the natural arms' selection.
Windows (36 tokens, as the short pairs' last 36 tokens): looping trace, its last 36 loop tokens (positions covered by
a 30-gram seen >= 20 times); non-looping trace, its last 36 thinking tokens before </think>.

select     choose the traces and windows; writes natural_traces.json (ids, windows, the prompts to drop from dev).
direction  v_l = mean residual (block output) over the looping windows minus over the non-looping windows, averaged
           per trace, unit norm. Writes v_n32.pt in loop_s_learn.py's format.
train      learned s for ul (unlikelihood on the looping windows' tokens) + beta * KL(unedited || edited) on the
           non-looping windows' tokens; Adam lr 3e-3 from s = 0, 24 full-batch updates, every update kept
           (s/ul_n32_b{beta}.pt with all_s, for loop_s_devloss.py score and loop_s_learn.py export).

  natural_direction.py select --bundle MATH_DEV_CONTROL --bundle LCB_DEV_CONTROL --stop-traces MATH_TR --stop-traces LCB_TR --out S
  natural_direction.py direction --out S --model M
  natural_direction.py train --out S --model M --beta 0.1
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import loop_pairs_edit as LPE  # noqa: E402
from loop_pairs_edit import decoder_layers, hidden_size, load_model, load_tokenizer, log, random_dirs, residuals  # noqa: E402
from probe.labeling import loop_token_positions  # noqa: E402

W, N_LOOP, N_FIN_PER_DS, SEED = 36, 32, 16, 20261111
LR, STEPS = 3e-3, 24
THINK_END = 151668  # </think> in the Qwen3 tokenizer (checked against the tokenizer in direction/train)


def cmd_select(a):
    from loop_s_devloss import load_traces
    stop = {}  # prompts whose traces the stopping rule scores, drawn exactly as loop_s_devloss.py draws them
    for f in a.stop_traces:
        ds = "math_level5" if "math_level5" in Path(f).name else "livecodebench"
        fin, lp = load_traces(f, THINK_END, None, 12)
        stop[ds] = {t["id"] for t in fin + lp}
    rng = random.Random(SEED)
    loop, fin, drop = {}, {}, {}
    for b in a.bundle:
        rows = [json.loads(l) for l in gzip.open(b, "rt")]
        ds = "math_level5" if "math_level5" in Path(b).name else "livecodebench"
        L, F = [], []
        for r in rows:
            if r["sample_id"] in stop[ds]:
                continue
            x = r["rollouts"][0]
            c = [int(t) for t in x["completion_token_ids"]]
            lp = loop_token_positions(c)
            t = {"dataset": ds, "sample_id": r["sample_id"], "prompt": [int(i) for i in r["prompt_token_ids"]], "completion_len": len(c)}
            if lp:
                t["window"] = lp[-W:]  # completion positions
                L.append(t)
            elif x["finish_reason"] == "stop" and THINK_END in c:
                e = c.index(THINK_END)
                if e >= W:
                    t["window"] = list(range(e - W, e))
                    F.append(t)
        rng.shuffle(L); rng.shuffle(F)
        loop[ds], fin[ds] = L, F[:N_FIN_PER_DS]
        log(ds, "looping traces available", len(L), "finished", len(F))
    ml = loop["math_level5"]
    lcb = loop["livecodebench"][: N_LOOP - len(ml)]
    chosen_loop = ml + lcb
    assert len(chosen_loop) == N_LOOP, len(chosen_loop)
    chosen_fin = fin["math_level5"] + fin["livecodebench"]
    for t in chosen_loop + chosen_fin:
        drop.setdefault(t["dataset"], []).append(t["sample_id"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"seed": SEED, "window": W, "bundles": [str(Path(b).name) for b in a.bundle],
            "n_loop": {ds: sum(t["dataset"] == ds for t in chosen_loop) for ds in loop},
            "n_fin": {ds: sum(t["dataset"] == ds for t in chosen_fin) for ds in fin},
            "drop_from_dev": drop}
    (out / "natural_traces.json").write_text(json.dumps({**meta, "loop": chosen_loop, "fin": chosen_fin}))
    (out / "natural_traces_summary.json").write_text(json.dumps(meta, indent=1))
    log(json.dumps({k: meta[k] for k in ("n_loop", "n_fin")}))


def load_sel(out):
    d = json.loads((Path(out) / "natural_traces.json").read_text())
    return d["loop"], d["fin"]


def full_ids(t, bundle_rows):
    return t["prompt"] + bundle_rows[(t["dataset"], t["sample_id"])]


def completions(a):
    rows = {}
    for b in a.bundle:
        for l in gzip.open(b, "rt"):
            r = json.loads(l)
            ds = "math_level5" if "math_level5" in Path(b).name else "livecodebench"
            rows[(ds, r["sample_id"])] = [int(t) for t in r["rollouts"][0]["completion_token_ids"]]
    return rows


def cmd_direction(a):
    out = Path(a.out)
    model, path = load_model()
    assert load_tokenizer(path).convert_tokens_to_ids("</think>") == THINK_END
    comp = completions(a)
    loop, fin = load_sel(out)
    B, D = LPE.BLOCKS, hidden_size(model)
    acc = {k: {l: torch.zeros(D, dtype=torch.float64) for l in B} for k in ("loop", "fin")}
    for key, ts in (("loop", loop), ("fin", fin)):
        for t in ts:
            ids = full_ids(t, comp)
            P = len(t["prompt"])
            last = P + max(t["window"])
            h = residuals(model, ids[: last + 1])
            idx = torch.tensor([P + i for i in t["window"]], device="cuda")
            for l in B:
                acc[key][l] += h[l][idx].mean(0).double().cpu()
    diff = {l: acc["loop"][l] / len(loop) - acc["fin"][l] / len(fin) for l in B}
    V = {l: (diff[l] / diff[l].norm()).float() for l in B}
    diag = {"model": Path(LPE.MODEL_ID).name, "blocks": list(B), "n_loop": len(loop), "n_fin": len(fin),
            "per_block": {l: {"diff_norm": float(diff[l].norm())} for l in B}}
    if a.ref_v:
        R = torch.load(a.ref_v)["learned"]
        for l in B:
            diag["per_block"][l]["cos_vs_short_v_n32"] = float(V[l] @ R[l])
    torch.save({"learned": V, "random": random_dirs(D), "n": 32, "model": LPE.MODEL_ID, "blocks": list(B),
                "source": "natural traces"}, out / "v_n32.pt")
    (out / "v_n32_diag.json").write_text(json.dumps(diag, indent=1))
    log(json.dumps(diag))


class Edit:
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


def window_logprobs(model, ids, pos):
    """log-probs predicting the tokens at absolute positions pos (from position p - 1), without full-vocab logits."""
    h = model.get_decoder()(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).last_hidden_state[0]
    return model.get_output_embeddings()(h[torch.tensor(pos, device="cuda") - 1]).float().log_softmax(-1)


def cmd_train(a):
    out = Path(a.out)
    (out / "s").mkdir(parents=True, exist_ok=True)
    name = f"ul_n32_b{a.beta:g}"
    V = {l: v.cuda() for l, v in torch.load(out / "v_n32.pt")["learned"].items()}
    model, path = load_model()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()  # HF checkpoints layers only in training mode; Qwen3 has no dropout, so outputs are unchanged
    comp = completions(a)
    loop, fin = load_sel(out)

    def item(t):
        ids = full_ids(t, comp)
        P = len(t["prompt"])
        pos = [P + i for i in t["window"]]
        return {"ids": ids[: max(pos) + 1], "pos": pos}

    L = [item(t) for t in loop]
    F = [item(t) for t in fin] if a.beta else []
    with torch.no_grad():
        base = [window_logprobs(model, it["ids"], it["pos"]) for it in F]
    S = {l: torch.zeros(hidden_size(model), device="cuda", requires_grad=True) for l in LPE.BLOCKS}
    opt = torch.optim.Adam(list(S.values()), lr=LR)
    edit = Edit(model, V, S)
    hist, all_s = [], [{l: S[l].detach().cpu().clone() for l in LPE.BLOCKS}]
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        tot = {"ul": 0.0, "kl": 0.0}
        for it in L:
            lp = window_logprobs(model, it["ids"], it["pos"])
            tg = torch.tensor([it["ids"][p] for p in it["pos"]], device="cuda")[:, None]
            p = lp.gather(1, tg).exp().clamp(max=1 - 1e-6)
            ul = -torch.log1p(-p).mean()
            (ul / len(L)).backward()
            tot["ul"] += float(ul) / len(L)
        for it, b in zip(F, base):
            lp = window_logprobs(model, it["ids"], it["pos"])
            kl = (b.exp() * (b - lp)).sum(-1).mean()
            (a.beta * kl / len(F)).backward()
            tot["kl"] += float(kl) / len(F)
        opt.step()
        all_s.append({l: S[l].detach().cpu().clone() for l in LPE.BLOCKS})
        h = {"step": step, "train_terms": tot, "s_norm": {l: float(S[l].norm()) for l in LPE.BLOCKS},
             "cos_s_v": {l: float(S[l] @ V[l] / (S[l].norm() + 1e-12)) for l in LPE.BLOCKS}}
        hist.append(h)
        log(name, "step", step, {k: round(v, 4) for k, v in tot.items()}, "|s|", [round(x, 3) for x in h["s_norm"].values()])
    edit.close()
    torch.save({"obj": "ul", "n": 32, "beta": a.beta, "best_step": STEPS, "s": all_s[-1], "v": {l: v.cpu() for l, v in V.items()},
                "history": hist, "all_s": all_s, "source": "natural traces"}, out / "s" / f"{name}.pt")
    (out / "s" / f"{name}.json").write_text(json.dumps({"name": name, "history": hist}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "direction", "train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", action="append", default=[], help="unedited dev rollout bundles (MATH, LiveCodeBench)")
    ap.add_argument("--stop-traces", action="append", default=[],
                    help="the bundles loop_s_devloss.py score reads (MATH, LiveCodeBench); their scored prompts are left out")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--ref-v", default="", help="the short pairs' v_n32.pt, to log the cosine with it")
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    LPE.configure(a.model or None, None, None)
    {"select": cmd_select, "direction": cmd_direction, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
