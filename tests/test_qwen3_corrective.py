"""Regression checks for the released Qwen3 Corrective NTP results."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Qwen3CorrectiveChecks(unittest.TestCase):
    def test_development_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "selection.json"
            subprocess.run([
                sys.executable, str(ROOT / "scripts/steer/exp1_select.py"),
                "--only", "clasd",
                "--rollouts", str(ROOT / "results/qwen3/rollouts_dev_clasd.json"),
                "--scores", str(ROOT / "results/qwen3/devscore_clasd.json"),
                "--json", str(out),
            ], check=True, capture_output=True)
            self.assertEqual(json.loads(out.read_text()),
                             json.loads((ROOT / "results/qwen3/selection_clasd.json").read_text()))

    def test_combined_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "summary.json"
            subprocess.run([
                sys.executable, str(ROOT / "scripts/report/qwen3_two_run_summary.py"),
                "--out", str(out), "--check",
            ], check=True, capture_output=True)
            self.assertEqual(json.loads(out.read_text()),
                             json.loads((ROOT / "results/qwen3/exp1_test_summary_with_clasd.json").read_text()))


if __name__ == "__main__":
    unittest.main()
