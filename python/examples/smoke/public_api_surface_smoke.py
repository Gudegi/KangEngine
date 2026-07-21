"""Validate canonical Python package paths without creating GPU resources."""

import importlib.util

import kangengine as ke
from kangengine._core import _ke
from kangengine.material import materials
from kangengine.motion_module import modules as motion_modules


def main() -> None:
    # Low-level render objects are the pybind types themselves, re-exported
    # under the stable public package path.
    assert ke.render.Renderer is _ke.Renderer
    assert ke.render.GraphicsDevice is _ke.GraphicsDevice
    assert ke.render.Shader is _ke.Shader
    assert ke.render.Texture is _ke.Texture
    assert not hasattr(ke.render, "NativeRenderer")
    assert not hasattr(ke.render, "NativeTexture")
    assert hasattr(ke.render.Texture, "width")
    assert hasattr(ke.render.Texture, "height")

    # Material facades are implemented inside their owning package while the
    # public class paths remain ke.material.*.
    assert ke.material.PBRMaterial is materials.PBRMaterial
    assert ke.material.PBRMaterial.__module__ == "kangengine.material"
    assert ke.material.NativePBRMaterial is _ke.PBRMaterial

    # Simulation and motion-module APIs have exclusive domain paths. Accessing
    # the packages may be lazy, but the classes must not reappear at ke.*.
    assert ke.sim.KangSimWorld.__module__ == "kangengine.sim"
    assert ke.sim.ControlMode.__module__ == "kangengine.sim"
    assert not hasattr(ke, "KangSimWorld")
    assert not hasattr(ke, "ControlMode")

    assert ke.motion_module.MotionModule.__module__ == "kangengine.motion_module"
    assert ke.motion_module.MotionModule is motion_modules.MotionModule
    assert importlib.util.find_spec("kangengine.motion_modules") is None
    assert not hasattr(ke, "MotionModule")
    assert not hasattr(ke, "ContactModule")

    print("PASS: canonical Python package API surface")


if __name__ == "__main__":
    main()
