"""Keep display-dependent wheel checks opt-in, including CUDA/OpenGL interop."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/wheel"))
import validate_wheel


class WheelValidationTests(unittest.TestCase):
    def test_display_checks_require_render_option(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = "kangengine-0.3.0-cp312-cp312-linux_x86_64.whl"
            runtime = "kangengine_physx_gpu-5.8.0-py3-none-linux_x86_64.whl"
            (root / "wheel-manifest.json").write_text(json.dumps({
                "platform": "linux", "wheels": [wheel, runtime],
            }))
            with zipfile.ZipFile(root / wheel, "w") as archive:
                archive.writestr(
                    "kangengine-0.3.0.dist-info/METADATA",
                    "Requires-Dist: nvidia-cuda-runtime==13.0.0\n",
                )

            for gpu, render in ((False, False), (True, False),
                                (False, True), (True, True)):
                args = ["validate_wheel.py", "--wheel-dir", directory]
                if gpu:
                    args.append("--gpu-smoke")
                if render:
                    args.append("--render-smoke")
                with self.subTest(gpu=gpu, render=render), \
                        patch.object(sys, "argv", args), \
                        patch.object(sys, "platform", "linux"), \
                        patch.object(validate_wheel, "inspect_wheel"), \
                        patch.object(validate_wheel, "create_validation_environment",
                                     return_value=(sys.executable, {})), \
                        patch.object(validate_wheel, "run") as run, \
                        redirect_stdout(io.StringIO()) as output:
                    validate_wheel.main()
                    scripts = {Path(call.args[2]).name for call in run.call_args_list}
                    for name in (
                        "scene_data_smoke.py",
                        "animation_sim_roundtrip_smoke.py",
                        "scene_resource_manager_smoke.py",
                        "render_component_lifecycle_smoke.py",
                        "sim_visual_batch_smoke.py",
                    ):
                        self.assertIn(name, scripts)
                    for name in (
                        "scene_renderer_boundary_smoke.py",
                        "scene_render_system_smoke.py",
                    ):
                        self.assertEqual(name in scripts, render, name)
                    self.assertEqual("mimickit_engine_gpu_reset_smoke.py" in scripts, gpu)
                    self.assertEqual("cuda_gl_visual_smoke.py" in scripts, gpu and render)
                    self.assertEqual("SKIP: rendering checks" in output.getvalue(), not render)


if __name__ == "__main__":
    unittest.main()
