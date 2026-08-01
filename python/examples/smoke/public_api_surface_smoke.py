"""Validate canonical Python package paths without creating GPU resources."""

import importlib.util

import kangengine as ke
from kangengine._core import _ke
from kangengine.app import application as app_application
from kangengine.material import materials
from kangengine.motion_module import editor as motion_editor
from kangengine.motion_module import modules as motion_modules
from kangengine.physics import wrappers as physics_wrappers
from kangengine.recording import VideoCaptureController, VideoRecorder
from kangengine.sim import sensor as sim_sensor
from kangengine.sim import run_mode as sim_run_mode
from kangengine.sim import timing as sim_timing
from kangengine.sim import world as sim_world
from kangengine.terrain import heightfield as terrain_heightfield


def main() -> None:
    assert ke.App is app_application.App
    assert ke.App.__module__ == "kangengine"
    assert hasattr(ke.App, "set_video_recording_resolution")
    assert not hasattr(ke, "NativeApp")
    assert "NativeApp" not in ke.app.__all__
    assert ke.SceneContext is app_application.SceneContext
    assert ke.SceneContext.__module__ == "kangengine"
    assert ke.DebugGeometry is app_application.DebugGeometry
    assert ke.DebugGeometry.__module__ == "kangengine"
    assert ke.DebugOverlay is app_application.DebugOverlay
    assert ke.DebugOverlay.__module__ == "kangengine"
    assert ke.RenderablePrimView is app_application.RenderablePrimView
    assert ke.RenderablePrimView.__module__ == "kangengine"
    for method_name in (
        "set_local_translation",
        "set_local_rotation",
        "set_local_rotation_axis_angle",
        "set_local_scale",
        "set_local_matrix",
        "set_world_translation",
        "set_world_rotation",
        "set_world_rotation_axis_angle",
        "set_world_matrix",
        "get_local_translation",
        "get_local_rotation",
        "get_world_translation",
        "get_world_rotation",
        "compute_local_matrix",
        "compute_world_matrix",
    ):
        assert hasattr(ke.RenderablePrimView, method_name)

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
    assert ke.motion_module.MotionEditor is motion_editor.MotionEditor
    assert ke.motion_module.MotionEditor.__module__ == "kangengine.motion_module"
    assert ke.motion_module.MotionPlayer.__module__ == "kangengine.motion_module"
    assert "MotionEditor" not in ke.__all__
    assert "MotionPlayer" not in ke.__all__
    assert importlib.util.find_spec("kangengine.motion_editor") is None
    assert importlib.util.find_spec("kangengine.motion_modules") is None
    assert ke.physics.PhysicsWorld is physics_wrappers.PhysicsWorld
    assert ke.physics.PhysicsWorld.__module__ == "kangengine.physics"
    assert ke.recording.VideoRecorder is VideoRecorder
    assert VideoRecorder.__module__ == "kangengine.recording.video_recorder"
    assert ke.recording.VideoCaptureController is VideoCaptureController
    assert ke.sim.KangSimWorld is sim_world.KangSimWorld
    assert ke.SimulationTimingConfig is sim_timing.SimulationTimingConfig
    assert ke.sim.SimulationTimingConfig is sim_timing.SimulationTimingConfig
    assert ke.SimulationTimingConfig.__module__ == "kangengine.sim"
    assert ke.SimulationRunMode is sim_run_mode.SimulationRunMode
    assert ke.SimulationRunConfig is sim_run_mode.SimulationRunConfig
    assert ke.SimulationPacer is sim_run_mode.SimulationPacer
    assert ke.sim.SimulationRunMode is sim_run_mode.SimulationRunMode
    assert ke.sim.KangSimWorld.__module__ == "kangengine.sim"
    assert ke.sim.ControlMode.__module__ == "kangengine.sim"
    assert ke.sim.ContactSensor is sim_sensor.ContactSensor
    assert ke.sim.ContactSensor.__module__ == "kangengine.sim"
    assert ke.sim.ContactSensorData.__module__ == "kangengine.sim"
    assert ke.sim.ForceSensor.__module__ == "kangengine.sim"
    assert "ContactSensor" not in ke.__all__
    assert "ForceSensor" not in ke.__all__
    assert importlib.util.find_spec("kangengine.sensor") is None

    # The optional CUDA implementation belongs to the native physics module.
    # This verifies namespace resolution and the CPU-build error path without
    # requiring a CUDA device or launching a kernel.
    native_aggregator = getattr(_ke.physics, "aggregate_contact_sensors_cuda", None)
    if native_aggregator is None:
        try:
            sim_sensor._contact_sensor_aggregator()
        except RuntimeError as error:
            assert "without CUDA contact aggregation" in str(error)
        else:
            raise AssertionError("CPU build unexpectedly resolved CUDA aggregation")
    else:
        assert sim_sensor._contact_sensor_aggregator() is native_aggregator

    assert ke.terrain.SubTerrain is terrain_heightfield.SubTerrain
    assert ke.terrain.SubTerrain.__module__ == "kangengine.terrain"
    assert ke.terrain.height_field_to_mesh is terrain_heightfield.height_field_to_mesh
    assert ke.terrain.height_field_to_mesh.__module__ == "kangengine.terrain"
    assert not hasattr(ke, "MotionModule")
    assert not hasattr(ke, "ContactModule")

    print("PASS: canonical Python package API surface")


if __name__ == "__main__":
    main()
