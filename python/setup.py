"""Setuptools compatibility hook for the prebuilt native extension.

CMake places ``_kangengine.so`` in the package before the wheel build.  Tell
setuptools that the distribution contains native code so the resulting wheel
is tagged for the active CPython ABI and platform instead of ``none-any``.
"""

from setuptools import Distribution, setup


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
