"""Guard the publication figure conventions against regeneration regressions."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NS = "{http://www.w3.org/2000/svg}"


class BlogFigureChecks(unittest.TestCase):
    def test_grid_styles_and_methods(self):
        files = list((ROOT / "blog/figures").glob("*delta_grid*.svg"))
        files = [p for p in files if "with_random" not in p.name]
        self.assertEqual(len(files), 8)
        for p in files:
            tree = ET.parse(p)
            texts = list(tree.iter(NS + "text"))
            strings = ["".join(t.itertext()) for t in texts]
            self.assertFalse(any("Contrast" in t or "†" in t for t in strings), p.name)
            for t, s in zip(texts, strings):
                if s == "Corrective NTP":
                    self.assertNotIn(t.get("font-weight"), ("700", "bold"), p.name)
                if re.match(r"^[+−-]\d", s):
                    self.assertEqual(t.get("style"), "fill:#000000", (p.name, s))
            if "with_positive_ntp" not in p.name:
                self.assertNotIn("Positive-only NTP", strings, p.name)
            self.assertEqual(strings.count("Corrective NTP"), 2, p.name)

    def test_regenerated_grids_match_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([
                sys.executable, str(ROOT / "scripts/report/build_blog_corrective_grids.py"),
                "--out", tmp,
            ], check=True, capture_output=True)
            generated = sorted(Path(tmp).glob("*.svg"))
            self.assertEqual(len(generated), 8)
            for p in generated:
                self.assertEqual(p.read_bytes().rstrip(),
                                 (ROOT / "blog/figures" / p.name).read_bytes().rstrip(),
                                 p.name)

    def test_notes_and_caption(self):
        s = (ROOT / "blog/main.md").read_text()
        notes = s.split("## 7. Notes", 1)[1]
        paragraphs = [p for p in notes.split("\n\n") if p.startswith("**")]
        self.assertTrue(paragraphs[0].startswith("**Reproduction details.**"))
        self.assertTrue(paragraphs[1].startswith("**Positive-only NTP.**"))
        self.assertNotIn("green: positive; red: loop/negative", s)
        self.assertIn("line at zero marks no change", s)
        self.assertNotIn("†", s)


if __name__ == "__main__":
    unittest.main()
