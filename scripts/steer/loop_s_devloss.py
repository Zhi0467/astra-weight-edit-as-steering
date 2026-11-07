#!/usr/bin/env python3
"""Teacher-forced dev scores of learned sensing vectors after every training update (experiment-1 early stopping).

The dev score should measure the behaviour we want, cheaply, teacher forced
through the edited model on the unedited model's own dev rollouts (whole traces, up to the generation cap):
  loop score      unlikelihood -log(1 - p) on loop tokens of looping traces. Loop tokens are the project's single
                  definition, probe.labeling.loop_token_positions (positions covered by a 30-gram seen >= 20 times in
                  the completion; the token-level form of the loop flag). Per-trace mean, averaged over looping traces.
  retention       next-token loss on every completion token that is not a loop token, in all traces (looping and not).
                  Per-trace mean, averaged over traces.
  closure         on finished non-looping traces: log p(</think>) at the position where the trace closes its thinking,
                  and the share of traces where </think> is still the edited model's top token there (greedy decoding
                  keeps closing exactly when it stays top).
The two scores are never added. Stopping rule (applied by --summarize): the update with the lowest loop score among
updates whose retention is within a tolerance of the unedited value (update 0) and whose closure top-1 share is not
below the unedited share. Several tolerances are reported; dev rollouts at updates 8-24 check the choice.

Traces: --bundle one or more unedited control rollout bundles on the dev split; --k looping and --k finished
non-looping traces per bundle, sampled with a fixed seed.

  loop_s_devloss.py score --out S_RUN --bundle B1 --bundle B2 --arms ul_n32_b0.1,clas_n4_b0 --json OUT.json
  loop_s_devloss.py summarize --json OUT.json [--tol 0.1,0.25,0.5]
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from probe.labeling import loop_token_positions  # noqa: E402

CHUNK = 2048
SEED = 20261106


def load_traces(path, think_id, end_id, k):
    fin, loop = [], []
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        x = r["rollouts"][0]
        p = [int(t) for t in r["prompt_token_ids"]]
        c = [int(t) for t in x["completion_token_ids"]]
        lp = loop_token_positions(c)
        if bool(lp) != bool(x["loop_flag"]):  # same definition as the generation-time flag
            print(f"warning: {r['sample_id']} loop_flag {x['loop_flag']} but {len(lp)} loop tokens", flush=True)
        t = dict(id=r["sample_id"], ids=p + c, p0=len(p), loop=set(lp))
        if lp:
            loop.append(t)
        elif x["finish_reason"] == "stop" and think_id in c:
            t["close"] = len(p) + c.index(think_id)
            fin.append(t)
    rng = random.Random(SEED)
    rng.shuffle(fin); rng.shuffle(loop)
    return fin[:k], loop[:k]


def cmd_score(a):
    import torch

    import loop_pairs_edit as LPE
    from loop_pairs_edit import decoder_layers, load_model, load_tokenizer

    class Edit:
        def __init__(self, model, V, S):
            self.handles = []
            for l in LPE.BLOCKS:
                v, s = V[l].cuda().float(), S[l].cuda().float()
                self.handles.append(decoder_layers(model)[l].mlp.register_forward_hook(self._hook(v, s)))

        @staticmethod
        def _hook(v, s):
            def f(_m, _i, o):
                xf = o.float()
                return (xf - (xf @ s)[..., None] * v).to(o.dtype)
            return f

        def close(self):
            for h in self.handles:
                h.remove()

    LPE.configure(a.model, a.blocks)
    model, path = load_model()
    tok = load_tokenizer(path)
    head = model.get_output_embeddings()
    body = model.get_decoder()  # decoder stack with final norm, without all-layer hidden states (long traces)
    think_id = tok.convert_tokens_to_ids("</think>")
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    traces = []
    for b in a.bundle:
        f, l = load_traces(b, think_id, end_id, a.k)
        print(f"{b}: {len(f)} finished (len {sorted(len(t['ids']) for t in f)}), {len(l)} looping "
              f"(len {sorted(len(t['ids']) for t in l)}, loop share "
              f"{[round(len(t['loop']) / (len(t['ids']) - t['p0']), 2) for t in l]})", flush=True)
        traces += f + l

    @torch.no_grad()
    def evaluate():
        r = {"loop_ul": [], "loop_lp": [], "ret": [], "ret_fin": [], "ret_loop": [], "close_lp": [], "close_top": []}
        for t in traces:
            h = body(input_ids=torch.tensor([t["ids"]], device="cuda"), use_cache=False).last_hidden_state[0]  # after final norm
            ids = torch.tensor(t["ids"], device="cuda")
            n = len(t["ids"]) - t["p0"]
            nll = torch.empty(n, device="cuda")
            for i in range(0, n, CHUNK):  # completion token i is predicted from position p0 + i - 1
                j = min(i + CHUNK, n)
                lp = head(h[t["p0"] + i - 1:t["p0"] + j - 1]).float().log_softmax(-1)
                nll[i:j] = -lp.gather(1, ids[t["p0"] + i:t["p0"] + j][:, None])[:, 0]
                if "close" in t and i <= t["close"] - t["p0"] < j:
                    q = t["close"] - t["p0"] - i
                    r["close_lp"].append(float(lp[q, think_id])); r["close_top"].append(float(int(lp[q].argmax()) == think_id))
            is_loop = torch.zeros(n, dtype=torch.bool, device="cuda")
            if t["loop"]:
                is_loop[torch.tensor(sorted(t["loop"]), device="cuda")] = True
                p = (-nll[is_loop]).exp().clamp(max=1 - 1e-6)
                r["loop_ul"].append(float(-torch.log1p(-p).mean())); r["loop_lp"].append(float(-nll[is_loop].mean()))
            ret = float(nll[~is_loop].mean())
            r["ret"].append(ret); r["ret_loop" if t["loop"] else "ret_fin"].append(ret)
        return {k: sum(v) / len(v) for k, v in r.items() if v}

    res = {"traces": [{"id": t["id"], "len": len(t["ids"]), "looping": bool(t["loop"])} for t in traces], "arms": {}}
    if Path(a.json).exists():
        res["arms"] = json.loads(Path(a.json).read_text()).get("arms", {})
    steps = [int(s) for s in a.steps.split(",")]
    base = None
    for name in a.arms.split(","):
        if name in res["arms"]:
            continue
        d = torch.load(Path(a.out) / "s" / f"{name}.pt")
        hist = []
        for st in steps:
            if st == 0 and base is not None:  # s = 0 is the unedited model for every arm
                m = dict(base)
            else:
                e = Edit(model, d["v"], d["all_s"][st])
                m = evaluate()
                e.close()
                if st == 0:
                    base = dict(m)
            m["step"] = st
            hist.append(m)
            print(f"{name:18s} step {st:2d} loop UL {m['loop_ul']:.4f} (lp {m['loop_lp']:.3f}) | retention {m['ret']:.4f} "
                  f"(fin {m['ret_fin']:.4f} loop {m['ret_loop']:.4f}) | </think> lp {m['close_lp']:.2f} top {m['close_top']:.2f}",
                  flush=True)
        res["arms"][name] = {"obj": d["obj"], "beta": float(d["beta"]), "history": hist}
        Path(a.json).write_text(json.dumps(res, indent=1))


def choose(hist, tol):
    h0 = hist[0]
    ok = [h for h in hist if h["ret"] <= h0["ret"] * (1 + tol) and h["close_top"] >= h0["close_top"] - 1e-9]
    return min(ok, key=lambda h: (h["loop_ul"], h["step"]))["step"]


def cmd_summarize(a):
    res = json.loads(Path(a.json).read_text())
    tols = [float(x) for x in a.tol.split(",")]
    print(f"{'arm':18s} " + " ".join(f"tol{t:g}" for t in tols) + " | step: loop UL / retention / </think> top")
    out = {}
    for name, r in res["arms"].items():
        h = r["history"]
        ch = {t: choose(h, t) for t in tols}
        out[name] = ch
        cells = " ".join(f"{h_['step']}:{h_['loop_ul']:.2f}/{h_['ret']:.3f}/{h_['close_top']:.2f}" for h_ in h if h_["step"] % 4 == 0)
        print(f"{name:18s} " + " ".join(f"{ch[t]:>{len(f'tol{t:g}')}}" for t in tols) + " | " + cells)
    if a.write:
        res["choices"] = {n: {str(t): s for t, s in c.items()} for n, c in out.items()}
        Path(a.json).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--bundle", action="append", required=True)
    s.add_argument("--arms", required=True)
    s.add_argument("--steps", default=",".join(str(i) for i in range(0, 25, 2)))
    s.add_argument("--k", type=int, default=12)
    s.add_argument("--model")
    s.add_argument("--blocks")
    s.add_argument("--json", required=True)
    m = sub.add_parser("summarize")
    m.add_argument("--json", required=True)
    m.add_argument("--tol", default="0.1,0.25,0.5,1.0")
    m.add_argument("--write", action="store_true")
    a = ap.parse_args()
    cmd_score(a) if a.cmd == "score" else cmd_summarize(a)


if __name__ == "__main__":
    main()
