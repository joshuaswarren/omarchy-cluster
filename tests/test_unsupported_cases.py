"""CL16: docs/unsupported-cases.md exists, covers the supported classes the
merged drill actually handles, and names the remaining unsupported cases."""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "unsupported-cases.md")


class TestUnsupportedCases(unittest.TestCase):
    def setUp(self):
        self.text = open(DOC).read()

    def test_doc_exists_with_both_sections(self):
        self.assertIn("## Covered", self.text)
        self.assertIn("## Unsupported", self.text)

    def test_covered_cases_match_merged_code(self):
        # the three classes drill.py/guard actually handle — if the code
        # loses one, this row rots
        self.assertRegex(self.text, r"[Ww]orker.*(loss|kill)")
        self.assertRegex(self.text, r"[Rr]estart.*[Rr]eplay|[Rr]eplay")
        self.assertRegex(self.text, r"[Ss]tops answering|[Nn]ode.*answer")

    def test_unsupported_cases_named(self):
        for term in ("failover", "reboot", "gateway", "joint",
                     "partition", "checkpoint|resume", "retry"):
            self.assertRegex(self.text, term)

    def test_references_live_files(self):
        for ref in re.findall(r"\]\(([^)#]+\.md)(#[^)]*)?\)", self.text):
            path = os.path.join(ROOT, "docs", ref[0])
            self.assertTrue(os.path.exists(path), ref[0])

    def test_readme_marks_the_matrix_merged(self):
        readme = open(os.path.join(ROOT, "README.md")).read()
        self.assertIn("unsupported-cases.md", readme)
        self.assertNotIn("unsupported-cases matrix (the former", readme
                         .replace("](docs/unsupported-cases.md)", ")"))


if __name__ == "__main__":
    unittest.main()
