"""Check smoke selection, source coverage, and subprocess failure reporting."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_smokes
from smoke_cases import CASES, MANUAL_CASES, SUITES, Case


class SmokeRunnerTests(unittest.TestCase):
    def test_every_source_is_registered_or_explicitly_manual(self):
        root = run_smokes.ROOT
        sources = {
            str(path.relative_to(root))
            for directory, pattern in (("python/examples/smoke", "*.py"),
                                       ("examples/smokes", "*.cpp"))
            for path in (root / directory).rglob(pattern)
        }
        registered = {case.source for case in CASES}
        self.assertEqual(sources, registered | MANUAL_CASES.keys())
        self.assertFalse(registered & MANUAL_CASES.keys())
        ids = {case.id for case in CASES}
        self.assertEqual(len(ids), len(CASES))
        for suite, members in SUITES.items():
            self.assertTrue(members, suite)
            self.assertFalse(set(members) - ids, suite)

    def test_selection_retains_variants_without_duplicate_execution(self):
        selected = run_smokes.select_cases(["render-component", "render"], [])
        ids = [case.id for case in selected]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(SUITES["render-component"]), 31)
        self.assertIn("py:shared_material_updates_smoke", ids)
        self.assertIn("py:shared_material_updates_smoke:scene-renderer", ids)
        self.assertIn("cpp:render_backend_smoke:app", ids)
        self.assertTrue(all(case.source.endswith(".py") for case in
                            run_smokes.select_cases(["render-component"], [])))

    def test_failures_skips_timeouts_and_missing_binaries_do_not_stop_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "python").mkdir()
            (root / "python/local_marker.py").write_text("value = 42\n")
            scripts = {
                "fail.py": "raise RuntimeError('failure detail')\n",
                "skip.py": "print('SKIP: optional device unavailable')\n",
                "timeout.py": "import time; time.sleep(10)\n",
                "pass.py": "import local_marker, sys; assert local_marker.value == 42; print(sys.executable)\n",
            }
            for name, content in scripts.items():
                (root / name).write_text(content)
            cases = [Case(name, (), timeout=0.1 if name == "timeout.py" else 10)
                     for name in scripts]
            cases.insert(1, Case("missing.cpp", ()))
            with redirect_stdout(io.StringIO()):
                status = run_smokes.run_cases(cases, root=root, build_dir=root, log_dir=root)
            self.assertEqual(status, 1)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual([result["status"] for result in summary["results"]],
                             ["FAIL", "NOT_RUN", "SKIP", "FAIL", "PASS"])
            self.assertIn("failure detail", (root / "py__fail.log").read_text())
            self.assertIn(sys.executable, (root / "py__pass.log").read_text())

    def test_list_does_not_need_engine_or_create_logs(self):
        root = run_smokes.ROOT
        before = set((root / "build/smoke-logs").glob("*"))
        result = subprocess.run(
            [sys.executable, "-S", str(root / "python/scripts/run_smokes.py"), "--list"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("MANUAL", result.stdout)
        self.assertIn("requires=", result.stdout)
        all_result = subprocess.run(
            [sys.executable, "-S", str(root / "python/scripts/run_smokes.py"),
             "--suite", "all", "--suite", "core", "--list"],
            capture_output=True, text=True, check=True,
        )
        ids = [line.split()[0] for line in all_result.stdout.splitlines()
               if line.startswith(("py:", "cpp:"))]
        self.assertEqual(ids, [case.id for case in CASES])
        self.assertNotIn("MANUAL", all_result.stdout)
        self.assertEqual(before, set((root / "build/smoke-logs").glob("*")))


if __name__ == "__main__":
    unittest.main()
