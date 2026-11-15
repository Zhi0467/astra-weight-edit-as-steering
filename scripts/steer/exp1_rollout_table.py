import json, gzip, glob, os, re, sys
B = sys.argv[1]  # run root holding gen_dev/ and graded_dev/
rows = {}
for ds in ("math_level5", "livecodebench"):
    root = f"{B}/gen_dev/{ds}" if ds == "math_level5" else f"{B}/graded_dev/{ds}"
    for d in sorted(glob.glob(f"{root}/*")):
        a = os.path.basename(d)
        if ds == "math_level5" and not os.path.exists(f"{d}/DONE"): continue
        if ds == "livecodebench" and not os.path.exists(f"{d}/GRADED"): continue
        m = re.match(r"(.*?)(?:_s(\d+))?$", a); arm, st = m.group(1), int(m.group(2) or 24)
        if arm.startswith("sv_") or arm == "control": st = 0 if arm == "control" else st
        f = [x for x in glob.glob(f"{d}/*.jsonl.gz") if "preexisting" not in x][0]
        n = c = cl = lp = cap = 0
        for l in gzip.open(f, "rt"):
            r = json.loads(l)["rollouts"][0]; n += 1
            g = r.get("grading", {}); c += bool(g.get("correct", 0) or g.get("passed", False)); cl += "</think>" in r["completion_text"]
            lp += bool(r["loop_flag"]); cap += bool(r["max_length_hit"])
        rows.setdefault((arm, st), {})[ds] = dict(n=n, acc=c / n, closed=cl / n, loop=lp / n, cap=cap / n)
print(json.dumps({f"{a}@{s}": v for (a, s), v in sorted(rows.items())}))
