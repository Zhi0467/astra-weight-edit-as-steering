#!/usr/bin/env python3
"""Stopping update of each learned s in Astra's regime: the lowest teacher-forced loop score (unlikelihood on loop
tokens) among updates whose </think> top-1 share is no worse than the unedited model's (update 0).

Input: the per-arm JSONs of astra_regime_devloss.py. Prints "<arm> <update>" per arm.

  astra_stop_rule.py DEVLOSS_DIR/*.json
"""

import json
import sys
from pathlib import Path


def stop(hist):
    h0 = hist[0]
    ok = [h for h in hist if h["close_top"] >= h0["close_top"] - 1e-9]
    return min(ok, key=lambda h: (h["ul"], h["step"]))["step"]


for p in sys.argv[1:]:
    d = json.loads(Path(p).read_text())
    print(d["arm"], stop(d["history"]))
