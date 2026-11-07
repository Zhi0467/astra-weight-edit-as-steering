#!/usr/bin/env python3
"""Experiment 2: ways of building the steering vector v, with sensing along it (s = alpha v), Qwen3-1.7B blocks 15-18.

Edit as in experiment 1 (loop_s_learn.py): on the MLP output x of each edited block, x <- x - v (s . x) with s = alpha v
and v unit norm, i.e. the weight edit W_down <- W_down - alpha outer(v, v @ W_down).

Data: the first N (= 32, fixed by experiment 1's dev selection) hand-written pairs of scripts/steer/loop_short_pairs.json,
the same texts as loop_s_learn.py direction. Per pair and block: the last 36 token states of the repetitive text (label 1)
and of the productive text (label 0), read from the residual stream after the block (default) or from the block's MLP
output (md_mlp only).

Constructions (all unit norm, sign chosen so v . (mean-difference direction) > 0):
  md_resid  mean over pairs of (mean of the 36 repetitive states - mean of the 36 productive states), residual stream.
            Must equal v_n32.pt "learned" (experiment 1's s = alpha v arm); checked, cos >= 0.9999.
  md_mlp    the same mean difference on the MLP output x.
  logreg    L2-regularized logistic regression on the residual states (sklearn lbfgs, states divided by one global scale
            for conditioning only); C from a log grid chosen by leave-one-pair-out cross-validated log loss; refit on all
            pairs; v = normalized weight vector.
  rfm       the repository's Laplace RFM (src/probe/rfm.py) with the recipe of scripts/cue_branch/clas_directions.py:
            states scaled to unit norm, bandwidth {1, 1.5, 10, 100} x ridge {1e-3, 1e-2, 1e-1}, 5 AGOP iterations each,
            (bandwidth, ridge, iteration) chosen by AUROC on held-out pairs (6 of 32, seeded); v = top eigenvector of M.
  pc1       top right singular vector of the N x d matrix of per-pair difference vectors (not centred).
  random    the seeded random unit vector already in v_n32.pt (random_dirs), the experiment-2 test control.
Diagnostics per block: cosine with md_resid; leave-one-pair-out projection AUROC (states of the held-out pair scored by
x . v, v rebuilt without that pair) for md_resid, logreg and pc1; RFM's held-out-pair AUROC; written to the json.

  loop_v_constructions.py build  --out S_RUN --astra-dir ASTRA_PAIRS [--n 32]      -> S_RUN/v_exp2_n32.pt, .json
  loop_v_constructions.py export --out S_RUN --cond md_mlp:2 [--cond rfm:3 ...]   -> S_RUN/checkpoints/vc_<c>_n32_a<alpha>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_pairs_edit as LPE  # noqa: E402
import loop_s_learn as LSL  # noqa: E402
from loop_pairs_edit import SHORT_LAST, decoder_layers, load_model, load_tokenizer, log  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONSTRUCTIONS = ("md_resid", "md_mlp", "logreg", "rfm", "pc1", "random")
C_GRID = [10.0 ** k for k in range(-4, 5)]
RFM_GRID = [(bw, reg) for bw in (1.0, 1.5, 10.0, 100.0) for reg in (1e-3, 1e-2, 1e-1)]
RFM_ITERS, RFM_HELD, SEED = 5, 6, 20261106


@torch.no_grad()
def states(model, ids):
    """{block: (residual after the block [T, d], MLP output [T, d])} over the last SHORT_LAST tokens, float32 cpu."""
    out, hs = {}, []
    for l in LPE.BLOCKS:
        hs.append(decoder_layers(model)[l].mlp.register_forward_hook(
            lambda _m, _i, o, l=l: out.__setitem__(("mlp", l), o[0, -SHORT_LAST:].float().cpu())))
        hs.append(decoder_layers(model)[l].register_forward_hook(
            lambda _m, _i, o, l=l: out.__setitem__(("res", l), (o[0] if isinstance(o, tuple) else o)[0, -SHORT_LAST:].float().cpu())))
    model(torch.tensor([ids], device="cuda"), use_cache=False)
    for h in hs:
        h.remove()
    return {l: (out[("res", l)], out[("mlp", l)]) for l in LPE.BLOCKS}


def unit(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return (x / x.norm()).float()


def signed(v, ref):
    return v if float(v @ ref) >= 0 else -v


def md(R, P):  # R, P: [N, T, d]
    return unit((R.double().mean(1) - P.double().mean(1)).mean(0))


def pc1(R, P):
    D = (R.double().mean(1) - P.double().mean(1))  # [N, d], not centred
    return unit(torch.linalg.svd(D, full_matrices=False).Vh[0])


def logreg(R, P, C=None):
    """(unit weight vector, chosen C, cv log loss per C). C=None: choose by leave-one-pair-out CV log loss."""
    from sklearn.linear_model import LogisticRegression
    N, T, d = R.shape
    X = torch.cat([R, P], 1).reshape(-1, d).double().numpy()  # pair-major: pair i rows [2T*i, 2T*(i+1))
    y = np.tile(np.r_[np.ones(T), np.zeros(T)], N)
    g = np.repeat(np.arange(N), 2 * T)
    scale = float(np.median(np.linalg.norm(X, axis=1)))
    X = X / scale
    fit = lambda c, m: LogisticRegression(C=c, max_iter=5000, tol=1e-6).fit(X[m], y[m])
    cv = {}
    if C is None:
        for c in C_GRID:
            ll = 0.0
            for i in range(N):
                m = g != i
                p = np.clip(fit(c, m).predict_proba(X[~m])[:, 1], 1e-12, 1 - 1e-12)
                ll -= float(np.mean(y[~m] * np.log(p) + (1 - y[~m]) * np.log(1 - p)))
            cv[c] = ll / N
        C = min(cv, key=cv.get)
    return unit(fit(C, np.ones(len(y), bool)).coef_[0]), C, cv


def load_rfm():
    import importlib.util
    spec = importlib.util.spec_from_file_location("probe_rfm", REPO / "src/probe/rfm.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def rfm(R, P, seed=SEED):
    """(top eigenvector of the chosen M, info). Held-out pairs (RFM_HELD of N, seeded) pick bandwidth/ridge/iteration."""
    from sklearn.metrics import roc_auc_score
    mod = load_rfm()
    N, T, d = R.shape
    X = torch.cat([R, P], 1).reshape(-1, d).float()
    X = X / X.norm(dim=1, keepdim=True)
    y = np.tile(np.r_[np.ones(T), np.zeros(T)], N).astype(np.float32)
    held = set(np.random.default_rng(seed).choice(N, size=min(RFM_HELD, N - 1), replace=False).tolist())
    va = np.repeat(np.array([i in held for i in range(N)]), 2 * T)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xtr, xva = X[~va].to(dev), X[va].to(dev)
    ytr = torch.from_numpy(y[~va] - 0.5).to(dev)
    best = None
    for bw, reg in RFM_GRID:
        m = mod.LaplaceRFM(mod.LaplaceRFMConfig(bandwidth=bw, reg=reg, device=dev))
        try:
            m.fit_predictor(xtr, ytr)
        except torch.linalg.LinAlgError:
            continue
        for it in range(1, RFM_ITERS + 1):
            m.fit_M(xtr)
            try:
                m.fit_predictor(xtr, ytr)
            except torch.linalg.LinAlgError:
                break
            auc = float(roc_auc_score(y[va], m.predict(xva).cpu().numpy()))
            if best is None or auc > best["auc"]:
                best = {"auc": auc, "bandwidth": bw, "reg": reg, "iteration": it, "M": m.M.detach().cpu().clone()}
    ev, evec = torch.linalg.eigh(best["M"].double())
    info = {k: best[k] for k in ("auc", "bandwidth", "reg", "iteration")}
    info["top_eig_share"] = float(ev[-1] / ev.sum())
    return unit(evec[:, -1]), info


def proj_auroc_lopo(R, P, build):
    """Leave-one-pair-out AUROC of x . v, v rebuilt without the held-out pair (pooled over held-out pairs)."""
    from sklearn.metrics import roc_auc_score
    N = R.shape[0]
    sc, ys = [], []
    for i in range(N):
        m = torch.arange(N) != i
        v = signed(build(R[m], P[m]), md(R[m], P[m]))
        sc += (R[i] @ v).tolist() + (P[i] @ v).tolist()
        ys += [1] * R.shape[1] + [0] * P.shape[1]
    return float(roc_auc_score(ys, sc))


def cmd_build(a):
    out = Path(a.out)
    model, path = load_model()
    tok = load_tokenizer(path)
    pairs = LSL.load_pairs(a.astra_dir)[: a.n]
    S = {l: {"R": [], "P": [], "Rm": [], "Pm": []} for l in LPE.BLOCKS}
    for prod, rep in pairs:
        sr, sp = states(model, LSL.encode(tok, rep)[0]), states(model, LSL.encode(tok, prod)[0])
        for l in LPE.BLOCKS:
            S[l]["R"].append(sr[l][0]); S[l]["Rm"].append(sr[l][1])
            S[l]["P"].append(sp[l][0]); S[l]["Pm"].append(sp[l][1])
    del model
    torch.cuda.empty_cache()
    ref = torch.load(out / f"v_n{a.n}.pt")
    V = {c: {} for c in CONSTRUCTIONS}
    diag = {"n": a.n, "blocks": list(LPE.BLOCKS), "model": LPE.MODEL_ID, "per_block": {}}
    for l in LPE.BLOCKS:
        R, P = torch.stack(S[l]["R"]), torch.stack(S[l]["P"])
        Rm, Pm = torch.stack(S[l]["Rm"]), torch.stack(S[l]["Pm"])
        base = md(R, P)
        cos_ref = float(base @ ref["learned"][l])
        assert cos_ref > 0.9999, f"md_resid does not reproduce v_n{a.n}.pt at block {l}: cos {cos_ref}"
        V["md_resid"][l] = ref["learned"][l].float()
        V["md_mlp"][l] = md(Rm, Pm)  # a direction in the same space (the MLP output is added to the residual stream)
        w, C, cv = logreg(R, P)
        V["logreg"][l] = signed(w, base)
        vr, rinfo = rfm(R, P)
        V["rfm"][l] = signed(vr, base)
        V["pc1"][l] = signed(pc1(R, P), base)
        V["random"][l] = ref["random"][l].float()
        d = {"cos_md_resid": {c: float(V[c][l] @ base) for c in CONSTRUCTIONS},
             "logreg_C": C, "logreg_cv_logloss": {str(k): v for k, v in cv.items()}, "rfm": rinfo,
             "state_norm_median": float(torch.cat([R, P]).norm(dim=-1).median())}
        d["lopo_proj_auroc"] = {  # rfm omitted: refitting its grid 32 times is slow; its held-out AUROC is in "rfm"
            "md_resid": proj_auroc_lopo(R, P, md),
            "pc1": proj_auroc_lopo(R, P, pc1),
            "logreg": proj_auroc_lopo(R, P, lambda r, p: logreg(r, p, C)[0]),  # C fixed at the full-data choice
        }
        diag["per_block"][l] = d
        log("block", l, json.dumps(d))
    torch.save({"v": V, "n": a.n, "model": LPE.MODEL_ID, "blocks": list(LPE.BLOCKS)}, out / f"v_exp2_n{a.n}.pt")
    (out / f"v_exp2_n{a.n}.json").write_text(json.dumps(diag, indent=1))
    log("wrote", out / f"v_exp2_n{a.n}.pt")


@torch.no_grad()
def cmd_export(a):
    out = Path(a.out)
    d = torch.load(out / f"v_exp2_n{a.n}.pt")
    model, path = load_model()
    for cond in a.cond:
        c, alpha = cond.split(":")
        assert c in CONSTRUCTIONS, c
        V = d["v"][c]
        name = f"vc_{c}_n{a.n}_a{alpha}"
        dest = out / "checkpoints" / name
        saved, rel = {}, {}
        for l in LPE.BLOCKS:
            W = decoder_layers(model)[l].mlp.down_proj.weight
            saved[l] = W.detach().clone()
            v = V[l].to(W.device, torch.float32)
            E = W.float() - torch.outer(v, float(alpha) * v @ W.float())
            rel[l] = float((E - W.float()).norm() / W.float().norm())
            W.copy_(E.to(W.dtype))
        model.save_pretrained(dest, safe_serialization=True)
        for l, w in saved.items():
            decoder_layers(model)[l].mlp.down_proj.weight.copy_(w)
        for f in os.listdir(path):
            if f.startswith("tokenizer") or f in ("vocab.json", "merges.txt", "generation_config.json"):
                shutil.copy(os.path.join(path, f), dest / f)
        (dest / "EDIT.json").write_text(json.dumps({"base": LPE.MODEL_ID, "blocks": list(LPE.BLOCKS), "construction": c,
                                                    "alpha": float(alpha), "n_pairs": a.n, "relative_change": rel,
                                                    "formula": "W_down - outer(v, alpha * v @ W_down), v unit norm"}))
        log("exported", dest, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "export"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--astra-dir", default="")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--cond", action="append", default=[])
    ap.add_argument("--model", default="", help="HF id or local path (default Qwen/Qwen3-1.7B)")
    ap.add_argument("--revision", default="")
    a = ap.parse_args()
    LPE.configure(a.model or None, None, a.revision or None)
    {"build": cmd_build, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
