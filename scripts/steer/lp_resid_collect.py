#!/usr/bin/env python3
"""collect_model_stats.py with a run-time residual-stream projection at chosen Qwen3 blocks (vLLM 0.15).

LP_RESID=<spec.pt> from loop_pairs_edit.py resid-spec: {"blocks", "u": {block: unit vector}, "alpha"}.
vLLM keeps a block's output split as (hidden, residual) and adds them inside the next norm, so the block output is
h = hidden + residual. At each listed block: hidden <- hidden - alpha * u (u . h), i.e. h <- h - alpha * u u^T h.
Run with VLLM_DISABLE_COMPILE_CACHE=1 so no compiled graph is shared with unpatched runs.
The patch is installed at import, so the spawned data-parallel workers (which re-import this file) get it too.
Arguments are passed through to collect_model_stats.py unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rollout"))


def _install(spec_path: str) -> None:
    import vllm.model_executor.models.qwen3 as q

    spec = torch.load(spec_path)
    U = {int(l): v for l, v in spec["u"].items()}
    alpha = float(spec["alpha"])
    init0, fwd0 = q.Qwen3DecoderLayer.__init__, q.Qwen3DecoderLayer.forward

    def init(self, *args, prefix: str = "", **kw):
        init0(self, *args, prefix=prefix, **kw)
        l = int(q.extract_layer_index(prefix))
        self._lp_on = l in U
        if self._lp_on:
            # created under vLLM's device context, so it lands on the model's GPU
            self.register_buffer("_lp_u", torch.empty(U[l].numel(), dtype=torch.float32), persistent=False)
            self._lp_u.copy_(U[l])
            # alpha as a buffer too, so a compiled graph never carries it as a constant
            self.register_buffer("_lp_alpha", torch.full((), alpha, dtype=torch.float32), persistent=False)

    def forward(self, positions, hidden_states, residual):
        hidden_states, residual = fwd0(self, positions, hidden_states, residual)
        if self._lp_on:
            u = self._lp_u
            c = (hidden_states.float() + residual.float()) @ u
            hidden_states = (hidden_states.float() - self._lp_alpha * c[:, None] * u).to(hidden_states.dtype)
        return hidden_states, residual

    q.Qwen3DecoderLayer.__init__ = init
    q.Qwen3DecoderLayer.forward = forward
    print(f"[lp_resid] blocks {sorted(U)} alpha {alpha} from {spec_path}", flush=True)


_install(os.environ["LP_RESID"])

if __name__ == "__main__":
    import collect_model_stats

    collect_model_stats.main()
