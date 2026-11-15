#!/usr/bin/env python3
"""Stopping update of each learned sensing vector in Astra's regime, teacher-forced on the unedited model's dev traces.

After every training update, the arm's own objective and the </think> closure check are computed on the unedited
model's dev traces. The stopping rule (astra_stop_rule.py) keeps the lowest loop score (ul below) among updates whose
</think> top-1 share is no worse than the unedited model's. (`chosen_step` in the output is the lowest own-objective
update, an earlier rule kept only as a diagnostic.)
  traces    the unedited Qwen3.5-0.8B's own validation_1 rollouts (astra_regime.py gen, Astra's sampler). Finished = stopped,
            closed its thinking and not loop-flagged (all of them); looping = loop-flagged (a seeded sample of --k).
            Prompt = Astra's raw prompt; completion = the generated text re-tokenized.
  ntp       next-token loss on the completion tokens of finished traces (per-trace mean, averaged over traces).
  kl        KL(unedited || edited) next-token distributions on the same tokens (per-trace mean, averaged).
  ul        unlikelihood -log(1 - p) on the loop tokens of looping traces (positions covered by a 30-gram seen >= 20 times,
            probe.labeling.loop_token_positions; per-trace mean, averaged).
  objective clas = ntp + beta kl; ul = ul + beta kl; contrast = ntp + ul + beta kl (the training objectives' form).
Also logged per update (diagnostic, not used by the rule): log p(</think>) at the position where each finished trace
closes its thinking, and the share of finished traces where </think> is still the top token there.

  astra_regime_devloss.py --s-run S --gen-dir G/validation_1/control --astra-src A --model M --arm ul_n32_b0.1 --json OUT
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
from astra_regime import PREFIX, SUFFIX, load_problems  # noqa: E402
from probe.labeling import loop_token_positions  # noqa: E402

CHUNK = 2048
SEED = 20261106


class Edit:
    def __init__(self, model, V, S):
        self.handles = []
        for l in LPE.BLOCKS:
            v, s = V[l].cuda().float(), S[l].cuda().float()
            self.handles.append(LPE.decoder_layers(model)[l].mlp.register_forward_hook(self._hook(v, s)))

    @staticmethod
    def _hook(v, s):
        def f(_m, _i, o):
            xf = o.float()
            return (xf - (xf @ s)[..., None] * v).to(o.dtype)
        return f

    def close(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s-run", required=True)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--astra-src", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--blocks", default="15,19")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--steps", default=",".join(str(i) for i in range(0, 25, 2)))
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    LPE.configure(a.model, a.blocks)
    model, path = LPE.load_model()
    tok = LPE.load_tokenizer(path)
    head, body = model.get_output_embeddings(), model.get_decoder()
    think_id = tok.convert_tokens_to_ids("</think>")
    P = load_problems(Path(a.astra_src))
    res = {r["id"]: r for r in map(json.loads, open(Path(a.gen_dir) / "results.jsonl"))}
    fin, loop = [], []
    for line in gzip.open(Path(a.gen_dir) / "texts.jsonl.gz", "rt"):
        x = json.loads(line)
        r = res[x["id"]]
        p = tok(PREFIX + P[x["id"]]["problem"] + SUFFIX, add_special_tokens=False)["input_ids"]
        c = tok(x["text"], add_special_tokens=False)["input_ids"]
        lp = loop_token_positions(c)
        t = dict(id=x["id"], ids=p + c, p0=len(p), loop=lp)
        if r["loop"] and lp:
            loop.append(t)
        elif not r["loop"] and r["finish"] == "stop" and think_id in c:
            t["close"] = len(p) + c.index(think_id)
            fin.append(t)
    rng = random.Random(SEED)
    rng.shuffle(loop)
    loop = loop[: a.k]
    print(f"{len(fin)} finished traces (len {sorted(len(t['ids']) for t in fin)}), {len(loop)} looping", flush=True)
    traces = fin + loop

    @torch.no_grad()
    def hidden(t):
        return body(input_ids=torch.tensor([t["ids"]], device="cuda"), use_cache=False).last_hidden_state[0]

    base_h = [hidden(t).cpu() for t in fin]  # unedited final hidden states of finished traces, for KL

    @torch.no_grad()
    def evaluate():
        r = {"ntp": [], "kl": [], "ul": [], "close_lp": [], "close_top": []}
        for ti, t in enumerate(traces):
            h = hidden(t)
            ids = torch.tensor(t["ids"], device="cuda")
            n = len(t["ids"]) - t["p0"]
            if "close" in t:
                h0 = base_h[ti].cuda()
                nll_s, kl_s = 0.0, 0.0
                for i in range(0, n, CHUNK):
                    j = min(i + CHUNK, n)
                    sl = slice(t["p0"] + i - 1, t["p0"] + j - 1)
                    lp = head(h[sl]).float().log_softmax(-1)
                    lp0 = head(h0[sl]).float().log_softmax(-1)
                    nll_s += float(-lp.gather(1, ids[t["p0"] + i:t["p0"] + j][:, None]).sum())
                    kl_s += float((lp0.exp() * (lp0 - lp)).sum())
                    if i <= t["close"] - t["p0"] < j:
                        q = t["close"] - t["p0"] - i
                        r["close_lp"].append(float(lp[q, think_id]))
                        r["close_top"].append(float(int(lp[q].argmax()) == think_id))
                r["ntp"].append(nll_s / n)
                r["kl"].append(kl_s / n)
            else:
                pos = torch.tensor(t["loop"], device="cuda") + t["p0"]
                us = []
                for i in range(0, len(pos), CHUNK):
                    pp = pos[i:i + CHUNK]
                    lp = head(h[pp - 1]).float().log_softmax(-1)
                    p_ = lp.gather(1, ids[pp][:, None])[:, 0].exp().clamp(max=1 - 1e-6)
                    us.append(-torch.log1p(-p_))
                r["ul"].append(float(torch.cat(us).mean()))
        return {k: sum(v) / len(v) for k, v in r.items() if v}

    d = torch.load(Path(a.s_run) / "s" / f"{a.arm}.pt")
    obj, beta = d["obj"], float(d["beta"])
    hist = []
    for st in [int(s) for s in a.steps.split(",")]:
        e = Edit(model, d["v"], d["all_s"][st])
        m = evaluate()
        e.close()
        k = beta * m["kl"]
        m["objective"] = {"clas": m["ntp"] + k, "ul": m["ul"] + k, "contrast": m["ntp"] + m["ul"] + k}[obj]
        m["step"] = st
        hist.append(m)
        print(f"{a.arm} step {st:2d} obj {m['objective']:.4f} ntp {m['ntp']:.4f} kl {m['kl']:.4f} ul {m['ul']:.4f} "
              f"</think> lp {m['close_lp']:.2f} top {m['close_top']:.2f}", flush=True)
    best = min(hist, key=lambda h: (h["objective"], h["step"]))
    out = {"arm": a.arm, "obj": obj, "beta": beta, "chosen_step": best["step"], "n_finished": len(fin), "n_looping": len(loop),
           "traces": [{"id": t["id"], "len": len(t["ids"]), "looping": "close" not in t} for t in traces], "history": hist}
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(a.arm, "chosen step", best["step"], flush=True)


if __name__ == "__main__":
    main()
