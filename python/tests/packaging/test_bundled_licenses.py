"""Check notice staging and coverage of embedded and repaired wheel libraries."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/wheel"))
from pack_wheel import stage_project
from wheel_licenses import (
    BUNDLE, INVENTORY, LIBRARY_PROJECTS, inspect_license_bundle, write_license_bundle,
)


class BundledLicenseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.compiled = self.root / "build/kangengine"
        self.destination = self.root / "stage"
        for directory in (self.source / "kangengine/licenses",
                          self.compiled / "licenses", self.destination):
            directory.mkdir(parents=True)
        for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md"):
            (self.source / name).write_text("source metadata")
        (self.source / "kangengine/__init__.py").write_text("")
        (self.source / "kangengine/licenses/openfbx.txt").write_text("wrong build")
        (self.compiled / "_kangengine.so").write_bytes(b"selected extension")
        self.notices = {
            "openfbx/LICENSE": b"selected OpenFBX notice \n",
            "tbb/LICENSE": b"selected TBB notice\n",
            "tbb/NOTICE": b"upstream attribution\n",
        }
        for name, content in self.notices.items():
            path = self.compiled / "licenses" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(content)
        (self.compiled / "licenses/manifest.txt").write_text("\n".join(self.notices) + "\n")
        (self.compiled / "licenses/versions.json").write_text(
            json.dumps({"openfbx": "1", "tbb": "2"})
        )
        (self.compiled / "licenses/stale.txt").write_text("not selected")

    def test_selected_build_preserves_license_and_notice_bytes(self):
        with patch("pack_wheel.sys.platform", "darwin"):
            stage_project(self.source, self.destination,
                          extension=self.compiled / "_kangengine.so")
        licenses = self.destination / "kangengine/licenses"
        self.assertEqual({p.name for p in licenses.iterdir()}, {INVENTORY, BUNDLE})
        with zipfile.ZipFile(licenses / BUNDLE) as archive:
            self.assertEqual(set(archive.namelist()), set(self.notices))
            for name, content in self.notices.items():
                self.assertEqual(archive.read(name), content)
        self.assertIn("tbb\t2\ttbb/NOTICE", (licenses / INVENTORY).read_text())

    def test_missing_build_notice_does_not_fall_back_to_source(self):
        (self.compiled / "licenses/openfbx/LICENSE").unlink()
        with patch("pack_wheel.sys.platform", "darwin"), self.assertRaises(FileNotFoundError):
            stage_project(self.source, self.destination,
                          extension=self.compiled / "_kangengine.so")

    def wheel(self, projects, binaries=(), *, damaged=None):
        sources = {}
        for project in projects:
            path = self.root / f"{project}.txt"
            path.write_text(f"{project} upstream license")
            sources[f"{project}/LICENSE"] = path
        bundle = self.root / "bundle"
        write_license_bundle(bundle, sources, {p: "1" for p in projects})
        if damaged is not None:
            with zipfile.ZipFile(bundle / BUNDLE, "w") as archive:
                for name, content in damaged.items():
                    archive.writestr(name, content)
        wheel = self.root / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name in (INVENTORY, BUNDLE):
                archive.write(bundle / name, f"kangengine/licenses/{name}")
            for binary in binaries:
                archive.writestr(binary, b"native fixture")
        return wheel

    def test_missing_or_empty_zipped_notice_fails(self):
        for damaged in ({}, {"openfbx/LICENSE": " \n"}):
            with self.subTest(damaged=damaged):
                wheel = self.wheel({"openfbx"}, damaged=damaged)
                with zipfile.ZipFile(wheel) as archive, self.assertRaises(AssertionError):
                    inspect_license_bundle(archive, "kangengine")

    def test_auditwheel_renamed_library_requires_its_notice(self):
        binary = "kangengine.libs/libtbb-01234567.so.12"
        wheel = self.wheel({"openfbx"}, [binary])
        with zipfile.ZipFile(wheel) as archive, self.assertRaisesRegex(AssertionError, "tbb"):
            inspect_license_bundle(archive, "kangengine")
        wheel = self.wheel({"openfbx", "tbb"}, [binary])
        with zipfile.ZipFile(wheel) as archive:
            inspect_license_bundle(archive, "kangengine")

    def test_unknown_bundled_library_fails(self):
        wheel = self.wheel({"openfbx"}, ["kangengine.libs/libunknown-abc.so.1"])
        with zipfile.ZipFile(wheel) as archive, self.assertRaisesRegex(AssertionError, "No license mapping"):
            inspect_license_bundle(archive, "kangengine")

    def test_static_dependency_requires_notice_without_separate_library(self):
        projects = LIBRARY_PROJECTS["_kangengine.so"] - {"libdeflate"}
        wheel = self.wheel(projects, ["kangengine/_kangengine.so"])
        with zipfile.ZipFile(wheel) as archive, self.assertRaisesRegex(AssertionError, "libdeflate"):
            inspect_license_bundle(archive, "kangengine")

    def test_usd_requires_its_embedded_dependencies(self):
        wheel = self.wheel({"OpenUSD"}, ["kangengine/_native/lib/libusd_usd.so"])
        with zipfile.ZipFile(wheel) as archive, self.assertRaisesRegex(AssertionError, "tbb"):
            inspect_license_bundle(archive, "kangengine")


if __name__ == "__main__":
    unittest.main()
