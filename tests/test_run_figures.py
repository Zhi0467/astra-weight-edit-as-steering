"""Fresh-run plumbing tests use small synthetic rollouts, not model generation."""
import contextlib
import gzip
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/report"))
import prepare_run_figures as p


class RunFigures(unittest.TestCase):
    def test_explicit_source_and_no_fallback(self):
        for args in ([], ["--run", "5"]):
            result = subprocess.run(["bash", str(ROOT / "run/figures.sh"), *args],
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                p.qwen(Path(d))

    def test_qwen_fresh_rollouts(self):
        import exp1_test_summary as t
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            base = ["control"] + [f"{fam}_n{n}_a1" if fam == "sv" else f"{fam}_n{n}_b0_s8"
                                 for n in (4, 32) for fam in ("sv", "ul", "clas")]
            corrective = [f"clasd_n{n}_b0.1_s16" for n in (4, 32)]
            p.write(work / "qwen3/exp1/exp1_selection.json", {"test_arms": base})
            p.write(work / "qwen3/exp1/selection_clasd.json", {"test_arms": ["control"] + corrective})
            labels = {(src, ds, i): bool(i) for src in t.SRCS for ds in t.DSS for i in (0, 1)}
            for arm in ["control_nothink"] + base + corrective:
                for src in t.SRCS:
                    for ds in t.DSS:
                        folder = work / "qwen3/runs/test" / src / ds / arm
                        folder.mkdir(parents=True)
                        (folder / ("DONE" if ds == "math_level5" else "GRADED")).touch()
                        with gzip.open(folder / "out.jsonl.gz", "wt") as f:
                            for i in (0, 1):
                                f.write(json.dumps({"sample_id": i, "rollouts": [{
                                    "grading": {"correct": arm.startswith("clasd")},
                                    "loop_flag": False, "completion_text": "</think>",
                                    "max_length_hit": False, "completion_token_count": 123
                                }]}) + "\n")
            with patch.object(t, "read_labels", return_value=labels), patch.object(t, "B", 20):
                result = p.read(p.qwen(work))
            self.assertEqual(result["levels"]["clasd_n4"]["all"]["tokens"]["mean"], 123)
            self.assertEqual(result["vs_control"]["clasd_n4"]["all"]["correct"]["mean"], 1)
            self.assertIn("sv_n32_a1", result["levels"])

    def test_astra_fresh_rollouts(self):
        import astra_regime_summary as t
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            selected = {f"{fam}_n{n}": f"{fam}_n{n}_a2" if fam == "sv" else f"{fam}_n{n}_b0"
                        for n in (4, 32) for fam in ("sv", "ul", "clas", "clasd")}
            for name in ("dev_selection", "dev_selection_clasd"):
                p.write(work / f"astra_regime/{name}.json", {"chosen_per_family_and_n": selected})
            for arm in ["control", "control_nothink"] + list(selected.values()):
                for split in t.CONF:
                    folder = work / "astra_regime/gen" / split / arm
                    folder.mkdir(parents=True)
                    rows = [{"id": f"{split}-{source}", "source": source,
                             **{key: 1 for _, key in t.METRICS}}
                            for source in ("gsm8k", "math")]
                    (folder / "results.jsonl").write_text("\n".join(map(json.dumps, rows)))
            with patch.object(t, "boot", return_value=[0, 0]), contextlib.redirect_stdout(io.StringIO()):
                result = p.read(p.astra(work))
            self.assertIn("clasd_n4 - sv_n4_a2", result["paired"])
            self.assertIn("ul_n32", result["per_arm"])
            self.assertNotIn("sv_n4_a4", result["per_arm"])


if __name__ == "__main__":
    unittest.main()
