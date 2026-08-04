"""Validate shared scene batches and ExternalBuffer component ownership."""

import gc
import weakref

import kangengine as ke
import torch


def _translate(prim, x, y, z):
    prim.set_local_translation(ke.vec3(float(x), float(y), float(z)))


def main():
    app = ke.App()
    app.initialize(width=64, height=64, hide_ui=True, headless=True)
    material = app.create_standard_materials().common
    render_system = app.get_scene_render_system()

    mesh = ke.scene.Prim.create_rectangle_data(1.0, 1.0, 1.0)
    first = app.scene.add_mesh("/shared/first", mesh, material)
    second = app.scene.add_mesh("/shared/second", mesh, material)
    _translate(first.prim, -2.0, 0.0, 0.0)
    _translate(second.prim, 2.0, 0.0, 0.0)
    second.prim.set_manipulation_policy(ke.scene.ManipulationPolicy.Self)

    assert render_system.registration_count == 2
    assert render_system.shares_batch(first.component, second.component)
    assert first.prim.get_path() != second.prim.get_path()

    first.set_visible(False)
    app.render_frame_once()
    hidden_pick = app.ray_pick(ke.vec3(-2.0, 0.0, 5.0), ke.vec3(0.0, 0.0, -1.0))
    second_pick = app.ray_pick(ke.vec3(2.0, 0.0, 5.0), ke.vec3(0.0, 0.0, -1.0))
    assert not hidden_pick.hit
    assert second_pick.hit
    assert second_pick.prim is second.prim
    assert (
        second_pick.prim.get_manipulation_policy() == ke.scene.ManipulationPolicy.Self
    )

    assert first.remove()
    assert render_system.registration_count == 1
    assert second.component.attached

    batch = app.scene.add_mesh(
        "/external/batch",
        mesh,
        material,
        transform_source=ke.render.TransformSource.ExternalBuffer,
    )
    transforms = torch.eye(4, dtype=torch.float32).repeat(3, 1, 1)
    transforms[:, 3, 0] = torch.tensor([-1.0, 0.0, 1.0])
    transforms_ref = weakref.ref(transforms)
    batch.set_transform_buffer(transforms)
    del transforms
    gc.collect()
    assert transforms_ref() is not None

    app.render_frame_once()
    batch_pick = app.ray_pick(ke.vec3(0.0, 0.0, 5.0), ke.vec3(0.0, 0.0, -1.0))
    assert batch_pick.hit
    assert batch_pick.transform_source == ke.render.TransformSource.ExternalBuffer
    assert batch_pick.prim is None

    assert batch.remove()
    gc.collect()
    assert transforms_ref() is None
    assert render_system.registration_count == 1
    assert second.remove()
    assert render_system.registration_count == 0

    print("PASS: scene instancing and ExternalBuffer component lifecycle")


if __name__ == "__main__":
    main()
