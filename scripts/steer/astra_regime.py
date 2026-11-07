#!/usr/bin/env python3
"""Learned vs scaled-v sensing in Astra's regime (Qwen3.5-0.8B, thinking on, sampled): reference edit, checks, generation.

Astra materials (github.com/RikiyaT/lm_understand-logs, exp8_math), copied under --astra-src:
  practice/{gsm8k_train,math_train}.jsonl  the practice problems with Astra's ids (gsm8k_train_<i> = openai/gsm8k main
                                           train row i; math_train_<subject>_<i> = EleutherAI/hendrycks_math <subject>
                                           train row i), gold answers already extracted (code/practice35.py)
  splits.json, fresh_splits.json           Astra's problem sets (validation_1 ... final_3)
  edit/{edit.npz,manifest.json}            Astra's submitted top-64 alpha-6 edit (layers 15 and 19 down_proj)
  final_confirmation.csv                   Astra's per-problem results on the 1,152 confirmation problems

astra-ckpt   Rebuild Astra's edited checkpoint bit-exactly from edit/ (sparse bf16 entries over the base tensors; the
             sha256 of the base and edited tensors is checked) into a copy of the base checkpoint.
astra-check  Compare our steering vectors v (v_n4.pt from loop_s_learn.py direction) with Astra's edit: the edit's
             change per layer is -6 outer(u, mask64 * (u @ W0)); report cos(v, u) via the change's top left singular
             vector, and the relative error of W0 - 6 outer(v, mask * (v @ W0)) against Astra's tensor.
gen          vLLM generation on Astra's problems with their evaluation protocol: the raw prompt
             '<|im_start|>user\\n' + problem + '\\nPlease reason step by step, and put your final answer within \\boxed{}.'
             + '<|im_end|>\\n<|im_start|>assistant\\n<think>\\n' (= the chat template with enable_thinking=True);
             temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, at most 32,768 new tokens, max_model_len
             32,768 + 2,048, one draw per problem with a seed fixed by the problem id (the same in every arm).
             Grading as Astra's code/mathqa.py: the last \\boxed{} (or \\fbox{}) after the last '</think>' (the whole
             text if there is none), brace matched; math-verify 0.9.0 verify(parse('\\boxed{gold}'), parse('\\boxed{pred}')).
             Loop flag: some 30-token sequence of the completion occurs at least 20 times.
             --sampler greedy|nopp: dev-only diagnostics (greedy without presence penalty; T 0.6 / top-p 0.95 /
             top-k 20 without presence penalty), same cap.
             Writes <out>/results.jsonl (per problem, no text), texts.jsonl.gz and summary.json.

random-ckpt  Random-direction control checkpoint (seeded unit r per layer from v_n32.pt "random"): --kind full
             (W0 - alpha r r^T W0) or --kind top64 (Astra's 64 columns only), --alpha, or --match CKPT: per-layer gain
             so the relative weight change ||dW||/||W0|| equals CKPT's (norm-matched control).
resid-spec   Spec for the run-time residual-stream version of an MLP-write edit (same v and s, read and written on the
             block output h): s = alpha v, or learned s from --s-file. gen --resid SPEC applies it with the base model.

  astra_regime.py random-ckpt --astra-src A --base P --v v_n32.pt --kind top64 --alpha 6 --dest D
  astra_regime.py resid-spec --astra-src A --v v_n32.pt [--alpha 4 | --s-file s/ul_n32_b0.1.pt] --dest spec.pt
  astra_regime.py gen ... --model BASE --resid spec.pt
  astra_regime.py astra-ckpt --astra-src A --base P --dest D
  astra_regime.py astra-check --astra-src A --base P --v V
  astra_regime.py gen --astra-src A --model M --split validation_4 --out O [--limit 8] [--gpu-mem 0.85]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

PREFIX = "<|im_start|>user\n"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768}
# dev-only diagnostic samplers (not part of the test comparison): greedy without presence penalty, and Astra's sampler
# without presence penalty
SAMPLERS = {"astra": SAMPLING, "greedy": {"temperature": 0.0, "max_tokens": 32768},
            "nopp": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0, "max_tokens": 32768}}
LOOP_N, LOOP_K = 30, 20
LAYERS = (15, 19)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tname(l):
    return f"model.language_model.layers.{l}.mlp.down_proj.weight"


# ---------------------------------------------------------------- problems
def load_problems(src: Path) -> dict:
    rows = {}
    for f in ("gsm8k_train.jsonl", "math_train.jsonl"):
        for line in open(src / "practice" / f):
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def split_ids(src: Path, split: str) -> list[str]:
    for f in ("splits.json", "fresh_splits.json"):
        d = json.loads((src / f).read_text())
        if split in d:
            return list(d[split])
    raise KeyError(split)


def problem_seed(pid: str) -> int:
    return int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- grading (Astra's code/mathqa.py)
_BOX_FORMS = re.compile(r"\\boxed|\\fbox")


def answer_text(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def _box_at(text, idx):
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = r[1:k].strip()
                    return out or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract(text):
    t = answer_text(text)
    return last_boxed(t), len(_BOX_FORMS.findall(t))


_PARSED = {}


def is_correct(pred, gold):
    import math_verify as mv

    if pred is None or gold is None:
        return False
    to = 5 if threading.current_thread() is threading.main_thread() else None

    def parsed(s):
        if s not in _PARSED:
            _PARSED[s] = mv.parse("\\boxed{" + s + "}", parsing_timeout=to)
        return _PARSED[s]

    g, p = parsed(gold), parsed(pred)
    if not g or not p:
        return False
    try:
        return bool(mv.verify(g, p, timeout_seconds=to))
    except Exception:  # noqa: BLE001
        return False


def loop_flag(ids, n=LOOP_N, k=LOOP_K) -> bool:
    if len(ids) < n * k:
        return False
    c = collections.Counter(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))
    return max(c.values()) >= k


# ---------------------------------------------------------------- Astra's edit
def _bf16_bits(t):
    import torch

    return t.contiguous().view(torch.int16).numpy().view("uint16")


def astra_tensors(src: Path, base_path: str) -> dict:
    """{name: (base bf16 tensor, edited bf16 tensor)} rebuilt from edit.npz, hashes checked (as apply_edit.py)."""
    import numpy as np
    import torch
    from safetensors.torch import load_file

    man = json.loads((src / "edit" / "manifest.json").read_text())
    z = np.load(src / "edit" / "edit.npz")
    wmap = json.loads((Path(base_path) / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for i, t in enumerate(man["tensors"]):
        assert t["form"] == "sparse", t["form"]
        W0 = load_file(str(Path(base_path) / wmap[t["name"]]))[t["name"]]
        b = _bf16_bits(W0)
        assert hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest() == t["base_sha256"], t["name"]
        e = b.copy().reshape(-1)
        e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        e = e.reshape(b.shape)
        assert hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest() == t["edited_sha256"], t["name"]
        E = torch.from_numpy(e.view("int16").copy()).view(torch.bfloat16)
        out[t["name"]] = (W0, E)
    return out


def cmd_astra_ckpt(a):
    from safetensors.torch import load_file, save_file

    src, base, dest = Path(a.astra_src), Path(a.base), Path(a.dest)
    T = astra_tensors(src, str(base))
    wmap = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {wmap[n] for n in T}
    dest.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(base):
        if (base / f).is_file() and f not in shards:
            shutil.copy(base / f, dest / f)
    rel = {}
    for shard in shards:
        sd = load_file(str(base / shard))
        for n, (W0, E) in T.items():
            if n in sd:
                sd[n] = E.contiguous()
                rel[n] = float((E.float() - W0.float()).norm() / W0.float().norm())
        save_file(sd, str(dest / shard), metadata={"format": "pt"})
    (dest / "EDIT.json").write_text(json.dumps({"base": str(base), "cond": "astra_top64_a6", "relative_change": rel,
                                                "source": "lm_understand-logs exp8_math/runs/p50c/edit (sha256 checked)",
                                                "formula": "W0 - 6 outer(u, mask64 * (u @ W0)), layers 15 and 19"}))
    log("astra checkpoint", dest, rel)


def cmd_astra_check(a):
    import torch

    T = astra_tensors(Path(a.astra_src), a.base)
    V = torch.load(a.v)["learned"]
    res = {}
    for l in LAYERS:
        W0, E = T[tname(l)]
        W0f, Ef = W0.float(), E.float()
        dW = Ef - W0f
        cols = (dW.abs().sum(0) > 0)
        U, S, _ = torch.linalg.svd(dW.double(), full_matrices=False)
        u = U[:, 0].float()
        v = V[l].float()
        mask = cols.float()
        R = (W0f - 6 * torch.outer(v, mask * (v @ W0f))).to(torch.bfloat16).float()
        res[l] = {"changed_columns": int(cols.sum()), "sv_ratio_2_to_1": float(S[1] / S[0]),
                  "abs_cos_v_u": float((v @ u).abs() / v.norm()),
                  "rel_err_ours_vs_astra": float((R - Ef).norm() / dW.norm()),
                  "frac_entries_bitwise_equal": float((R == Ef).float().mean())}
    log(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- generation
def cmd_gen(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(a.astra_src)
    P = load_problems(src)
    ids = split_ids(src, a.split)
    if a.limit:
        ids = ids[: a.limit]
    done_path = out / "summary.json"
    if done_path.exists():
        log("already done", out)
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    instr = SUFFIX.split("<|im_end|>")[0]
    if a.thinking == "on":
        prompts = [PREFIX + P[i]["problem"] + SUFFIX for i in ids]
        tmpl = tok.apply_chat_template([{"role": "user", "content": P[ids[0]]["problem"] + instr}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=True)
        assert tmpl == prompts[0], ("chat template differs from the raw prompt", tmpl[-120:], prompts[0][-120:])
    else:  # no-thinking reference row: same user turn, the template's enable_thinking=False (empty think block)
        prompts = [tok.apply_chat_template([{"role": "user", "content": P[i]["problem"] + instr}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in ids]
        assert prompts[0].startswith(PREFIX + P[ids[0]]["problem"] + instr), prompts[0][:200]
        log("no-thinking prompt tail:", repr(prompts[0][-80:]))
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
              max_model_len=SAMPLING["max_tokens"] + 2048, seed=0, max_num_seqs=a.max_num_seqs)
    samp = SAMPLERS[a.sampler]
    sps = [SamplingParams(n=1, seed=problem_seed(i), **samp) for i in ids]
    t0 = time.time()
    outs = llm.generate(prompts, sps)
    gen_s = time.time() - t0
    bad = sum(1 for o in outs if o.outputs[0].finish_reason not in ("stop", "length"))
    if bad:
        raise RuntimeError(f"{bad} generations ended without finishing or reaching max_tokens")
    rows = []
    with gzip.open(out / "texts.jsonl.gz", "wt") as ft:
        for pid, o in zip(ids, outs):
            s = o.outputs[0]
            pred, nbox = extract(s.text)
            r = {"id": pid, "source": P[pid]["source"], "gold": P[pid]["answer"], "pred": pred if pred is None or len(pred) <= 200 else pred[:200],
                 "boxes": nbox, "correct": bool(is_correct(pred, P[pid]["answer"])), "finish": s.finish_reason,
                 "cut": s.finish_reason == "length", "tokens": len(s.token_ids), "loop": loop_flag(list(s.token_ids)),
                 "think_closed": "</think>" in s.text, "seed": problem_seed(pid)}
            rows.append(r)
            ft.write(json.dumps({"id": pid, "text": s.text}) + "\n")
    with open(out / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    summ = {"model": a.model, "split": a.split, "n": n, "accuracy": sum(r["correct"] for r in rows) / n,
            "think_closed_share": sum(r["think_closed"] for r in rows) / n,
            "cut_share": sum(r["cut"] for r in rows) / n, "loop_share": sum(r["loop"] for r in rows) / n,
            "mean_tokens": sum(r["tokens"] for r in rows) / n, "gen_seconds": round(gen_s),
            "by_source": {s: {"n": len(g), "accuracy": sum(r["correct"] for r in g) / len(g)}
                          for s in ("gsm8k", "math") for g in [[r for r in rows if r["source"] == s]] if g},
            "protocol": {"prefix": PREFIX, "suffix": SUFFIX, "thinking": a.thinking, "sampler": a.sampler, "sampling": samp, "seed": "sha256(problem id)[:8] per problem",
                         "max_model_len": SAMPLING["max_tokens"] + 2048, "loop": [LOOP_N, LOOP_K]}}
    done_path.write_text(json.dumps(summ, indent=1))
    log(json.dumps({k: v for k, v in summ.items() if k != "protocol"}))


# ---------------------------------------------------------------- blog controls (Nov 7 2026)
def cmd_random_ckpt(a):
    """Random-direction control checkpoints: r = the seeded random unit vector per layer stored in --v (v_n32.pt
    "random", from loop_pairs_edit.random_dirs). --kind full: W0 - alpha outer(r, r @ W0) on every column (matches
    s = alpha v); --kind top64: W0 - alpha outer(r, mask * (r @ W0)) on Astra's 64 edited columns per layer (the
    columns Astra's edit changes; matches the astra_top64_a6 edit with u replaced by r)."""
    import torch
    from safetensors.torch import load_file, save_file

    base, dest = Path(a.base), Path(a.dest)
    R = torch.load(a.v)["random"]
    T = astra_tensors(Path(a.astra_src), str(base)) if a.kind == "top64" else None
    wmap = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    names = {tname(l): l for l in LAYERS}
    shards = {wmap[n] for n in names}
    dest.mkdir(parents=True, exist_ok=True)
    for f in os.listdir(base):
        if (base / f).is_file() and f not in shards:
            shutil.copy(base / f, dest / f)
    rel, gains = {}, {}
    for shard in shards:
        sd = load_file(str(base / shard))
        for n, l in names.items():
            if n not in sd:
                continue
            W0 = sd[n].float()
            r = R[l].float()
            r = r / r.norm()
            g = r @ W0
            if T is not None:
                g = g * ((T[n][1].float() - T[n][0].float()).abs().sum(0) > 0).float()
            alpha = a.alpha
            if a.match:  # norm-matched: per-layer gain so that ||dW|| / ||W0|| equals the target checkpoint's
                target = json.loads((Path(a.match) / "EDIT.json").read_text())["relative_change"][n]
                alpha = target * float(W0.norm()) / float(torch.outer(r, g).norm())
            E = (W0 - alpha * torch.outer(r, g)).to(sd[n].dtype)
            gains[n] = alpha
            rel[n] = float((E.float() - W0).norm() / W0.norm())
            sd[n] = E.contiguous()
        save_file(sd, str(dest / shard), metadata={"format": "pt"})
    (dest / "EDIT.json").write_text(json.dumps({"base": str(base), "cond": f"random_{a.kind}" + (f"_match:{a.match}" if a.match else f"_a{a.alpha:g}"), "relative_change": rel, "gain": gains,
                                                "direction": f"{a.v} random (seeded unit vector per layer)",
                                                "formula": "W0 - alpha outer(r, mask * (r @ W0))" + (" (Astra's 64 columns)" if T is not None else " (all columns)")}))
    log("random checkpoint", dest, rel)


def cmd_resid_spec(a):
    """Run-time residual-stream version of an MLP-write edit: same v and s, applied to the block output h instead of
    the block's MLP write x:  h <- h - v (s . h) at each edited layer. --s-file: learned s (s/<arm>.pt, kept step);
    otherwise s = alpha v."""
    import torch

    d = torch.load(a.v)
    V = {int(l): t.float() for l, t in d["learned"].items()}
    if a.s_file:
        S = {int(l): t.float() for l, t in torch.load(a.s_file)["s"].items()}
    else:
        S = {l: a.alpha * v for l, v in V.items()}
    torch.save({"blocks": sorted(V), "v": V, "s": S, "source": {"v": a.v, "s_file": a.s_file, "alpha": a.alpha}}, a.dest)
    log("resid spec", a.dest, {l: round(float(S[l].norm()), 3) for l in S})


def install_resid(spec_path: str) -> None:
    """Patch vLLM's Qwen3.5 decoder layer: at the spec's layers, hidden <- hidden - v (s . (hidden + residual)).
    vLLM keeps a block's output split as (hidden, residual), summed inside the next norm, so h = hidden + residual.
    Needs VLLM_ENABLE_V1_MULTIPROCESSING=0 (engine in this process) and VLLM_DISABLE_COMPILE_CACHE=1."""
    import torch
    import vllm.model_executor.models.qwen3_5 as q

    spec = torch.load(spec_path)
    V = {int(l): t for l, t in spec["v"].items()}
    S = {int(l): t for l, t in spec["s"].items()}
    cls = q.Qwen3_5DecoderLayer
    init0, fwd0 = cls.__init__, cls.forward

    def init(self, *args, **kw):
        init0(self, *args, **kw)
        self._lp_on = self.layer_idx in V
        if self._lp_on:  # created under vLLM's device context, so the buffers land on the model's GPU
            self.register_buffer("_lp_v", torch.empty(V[self.layer_idx].numel(), dtype=torch.float32), persistent=False)
            self.register_buffer("_lp_s", torch.empty(S[self.layer_idx].numel(), dtype=torch.float32), persistent=False)
            self._lp_v.copy_(V[self.layer_idx])
            self._lp_s.copy_(S[self.layer_idx])

    def forward(self, *args, **kw):
        hidden, residual = fwd0(self, *args, **kw)
        if self._lp_on:
            c = (hidden.float() + residual.float()) @ self._lp_s
            hidden = (hidden.float() - c[:, None] * self._lp_v).to(hidden.dtype)
        return hidden, residual

    cls.__init__, cls.forward = init, forward
    log(f"[resid] layers {sorted(V)} from {spec_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["astra-ckpt", "astra-check", "gen", "random-ckpt", "resid-spec"])
    ap.add_argument("--astra-src", required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--dest", default="")
    ap.add_argument("--v", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--split", default="validation_4")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--sampler", choices=sorted(SAMPLERS), default="astra")
    ap.add_argument("--thinking", choices=["on", "off"], default="on",
                    help="off: chat template with enable_thinking=False (no-thinking reference row); on: Astra's protocol")
    ap.add_argument("--kind", choices=["full", "top64"], default="full", help="random-ckpt: columns edited")
    ap.add_argument("--alpha", type=float, default=1.0, help="random-ckpt / resid-spec gain")
    ap.add_argument("--match", default="", help="random-ckpt: checkpoint dir whose EDIT.json relative_change to match per layer")
    ap.add_argument("--s-file", default="", help="resid-spec: learned s (s_run/s/<arm>.pt); default s = alpha v")
    ap.add_argument("--resid", default="", help="gen: residual-stream spec from resid-spec (run-time edit, base model)")
    a = ap.parse_args()
    if a.cmd == "gen" and a.resid:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        install_resid(a.resid)
    {"astra-ckpt": cmd_astra_ckpt, "astra-check": cmd_astra_check, "gen": cmd_gen,
     "random-ckpt": cmd_random_ckpt, "resid-spec": cmd_resid_spec}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
