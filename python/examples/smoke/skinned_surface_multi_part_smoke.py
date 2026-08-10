"""Validate single-object, multi-part SkinnedSurface behavior without a window."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import kangengine as ke


class _Prim:
    def __init__(self, path):
        self.path = path
        self.color = None

    def set_mesh_data(self, mesh):
        self.mesh = mesh
        self.mesh_component = SimpleNamespace(
            mesh_data=mesh,
            resource_handle=ke.scene.InvalidResourceHandle,
        )

    def get_mesh_component(self):
        return self.mesh_component

    def get_path(self):
        return self.path

    def set_display_color_alpha(self, color):
        self.color = color if isinstance(color, ke.Vec4) else ke.Vec4(*color)

    def get_display_color_alpha(self):
        return self.color or ke.Vec4(1.0, 1.0, 1.0, 1.0)


class _View:
    def __init__(self, prim, material=None):
        self.prim = prim
        self.material = material

    def update_skinning(self, matrices):
        self.matrices = matrices

    @property
    def path(self):
        return self.prim.get_path()

    def update_geometry(self, positions, normals):
        self.positions = np.array(positions, copy=True)
        self.normals = None if normals is None else np.array(normals, copy=True)

    def set_visible(self, visible):
        self.visible = visible

    def set_casts_shadow(self, casts_shadow):
        self.casts_shadow = casts_shadow

    def set_alpha_mode(self, mode, cutoff):
        self.alpha_mode = (mode, cutoff)

    def set_texture(self, texture, slot):
        if not hasattr(self, "textures"):
            self.textures = {}
        self.textures[slot] = texture

    def set_base_color(self, color):
        self.prim.set_display_color_alpha(color)

    def get_base_color(self):
        return self.prim.get_display_color_alpha()

    def remove(self):
        if getattr(self, "removed", False):
            return False
        self.removed = True
        return True


class _Scene:
    def define_prim(self, path, prim_type):
        return _Prim(path)


class _App:
    def __init__(self):
        self.scene = _Scene()
        self._device = _Device()
        self.textures = []
        self._textures_by_uri = {}
        self.mesh_resources = {}
        self._next_mesh_handle = 1
        self.pbr_materials = []

    def add_skinned_mesh(self, prim, material, skin):
        ke.App._ensure_prim_mesh_resource(self, prim)
        return _View(prim, material)

    def _register_mesh_resource(self, mesh, display_name=None, uri=None):
        handle = self._next_mesh_handle
        self._next_mesh_handle += 1
        self.mesh_resources[handle] = mesh
        return handle

    def get_renderer(self):
        return SimpleNamespace(device=lambda: self._device)

    def load_texture(self, path, *, flip=True):
        resolved = str(Path(path).resolve())
        texture = self._textures_by_uri.get(resolved)
        if texture is None:
            texture = self._device.create_texture(resolved, flip)
            self._textures_by_uri[resolved] = texture
            self.textures.append(texture)
        return texture

    def create_pbr_material(self, **kwargs):
        material = SimpleNamespace(**kwargs)
        self.pbr_materials.append(material)
        return material


class _Device:
    def __init__(self):
        self.created = []

    def create_texture(self, path, flip_vertically):
        texture = (path, flip_vertically)
        self.created.append(texture)
        return texture


def _mesh(name, texture_path=""):
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    normals = np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
    bone_indices = np.asarray([[0, -1, -1, -1]] * 3, dtype=np.int32)
    bone_weights = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float32)
    return SimpleNamespace(
        name=name,
        vertices=positions,
        normals=normals,
        bone_indices=bone_indices,
        bone_weights=bone_weights,
        bone_node_indices=[0],
        inverse_bind_matrices=np.eye(4, dtype=np.float32)[None, ...],
        mesh_data=SimpleNamespace(indices=[0, 1, 2], uvs=[]),
        materials=[
            SimpleNamespace(
                name="shared_material",
                diffuse_color=[1.0, 1.0, 1.0, 1.0],
                diffuse_texture_path=texture_path,
                normal_texture_path="",
            )
        ],
        primary_material_index=0,
    )


def main():
    body = ke.asset.SMPLModel.load(
        ke.asset.smpl.repository_smpl_model_path()
    ).create_body()
    texture_path = str(Path(__file__).resolve().parents[3] / "assets/textures/awesomeface.png")
    result = SimpleNamespace(
        motion=SimpleNamespace(skeleton_tree=body.skeleton_tree),
        skinned_meshes=[
            _mesh("body", texture_path),
            _mesh("clothes", texture_path),
        ],
    )
    app = _App()
    surface = ke.visual.SkinnedSurface.create_from_fbx_result(
        app, "/character", result, object()
    )
    assert len(surface.prims) == 2
    assert len(surface.assets) == 2
    for prim in surface.prims:
        handle = prim.get_mesh_component().resource_handle
        assert handle != ke.scene.InvalidResourceHandle
        assert app.mesh_resources[handle] is prim.mesh
    assert len(app._device.created) == 1
    assert len(app.textures) == 1
    assert all(0 in view.textures for view in surface.views)
    assert all(
        view.alpha_mode == (ke.render.AlphaMode.MASK, 0.5)
        for view in surface.views
    )

    imported_surface = ke.visual.SkinnedSurface.create_from_fbx_result(
        app, "/imported_character", result
    )
    assert len(app.pbr_materials) == 1
    assert imported_surface.views[0].material is imported_surface.views[1].material
    assert imported_surface.views[0].material.base_color_texture is app.textures[0]
    assert all(
        view.alpha_mode == (ke.render.AlphaMode.MASK, 0.5)
        for view in imported_surface.views
    )

    rotations = np.zeros((24, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    root = np.zeros(3, dtype=np.float32)
    state = ke.animation.SkeletonState.from_rotation_and_root_translation(
        body.skeleton_tree, rotations, root
    )
    surface.apply_state(state)
    surface.apply_state(root, rotations)
    assert all(view.matrices.shape == (1, 4, 4) for view in surface.views)

    ghost = surface.create_instance(
        "/character_ghost", color=(0.2, 0.6, 1.0, 0.25)
    )
    ghost.apply_state(state)
    assert ghost.assets == surface.assets
    assert all(
        ghost_part.prim.mesh is source_part.prim.mesh
        for ghost_part, source_part in zip(ghost.views, surface.views)
    )
    assert all(
        ghost_part.material is source_part.material
        for ghost_part, source_part in zip(ghost.views, surface.views)
    )
    assert all(
        view.alpha_mode == (ke.render.AlphaMode.BLEND, 0.5)
        for view in ghost.views
    )
    ghost.set_alpha(0.1)
    assert all(abs(view.get_base_color().w - 0.1) < 1e-6 for view in ghost.views)
    assert all(view.get_base_color().w == 1.0 for view in surface.views)
    assert ghost.remove()
    assert not ghost.remove()
    try:
        ghost.apply_state(state)
    except RuntimeError as error:
        assert "removed" in str(error)
    else:
        raise AssertionError("removed surface accepted a pose update")

    single_surface = ke.visual.SkinnedSurface.create_from_fbx(
        app,
        "/single_character",
        body.skeleton_tree,
        result.skinned_meshes[0],
        object(),
    )
    single_handle = single_surface.prim.get_mesh_component().resource_handle
    assert single_handle != ke.scene.InvalidResourceHandle
    assert app.mesh_resources[single_handle] is single_surface.prim.mesh

    source_vertices = np.asarray(single_surface.asset.positions).copy()
    deformed_vertices = source_vertices.copy()
    deformed_vertices[:, 2] += 0.25
    source_mesh_vertices = list(single_surface.prim.mesh.vertices)
    single_surface.update_bind_geometry(deformed_vertices)
    assert np.array_equal(single_surface.asset.positions, source_vertices)
    assert single_surface.prim.mesh.vertices == source_mesh_vertices
    assert np.array_equal(single_surface.view.positions, deformed_vertices)
    single_surface.reset_bind_geometry()
    assert np.array_equal(single_surface.view.positions, source_vertices)

    surface.set_visible(False)
    surface.set_casts_shadow(False)
    surface.set_alpha_mode(ke.render.AlphaMode.OPAQUE)
    assert all(not view.visible and not view.casts_shadow for view in surface.views)
    print("PASS: multi-part SkinnedSurface")


if __name__ == "__main__":
    main()
