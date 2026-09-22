"""Tag the native runtime by platform, without a CPython ABI dependency."""

from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel


class RuntimeDistribution(Distribution):
    def has_ext_modules(self):
        return True


class RuntimeWheel(bdist_wheel):
    def run(self):
        if not Path("kangengine_physx_gpu/lib/libPhysXGpu_64.so").is_file():
            raise RuntimeError(
                "Build this runtime through make wheel to stage the PhysX GPU binary"
            )
        super().run()

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(distclass=RuntimeDistribution, cmdclass={"bdist_wheel": RuntimeWheel})
