"""Exercise native and USD scene data without initializing a renderer or physics world."""

import gc
from pathlib import Path
import sys
import tempfile

import numpy as np
import kangengine as ke


def assert_no_viewer_or_physics():
    assert "kangengine.app.application" not in sys.modules
    assert "kangengine.physics" not in sys.modules


def native_scene():
    scene = ke.scene.create_backend(type=ke.scene.BackendType.NATIVE)
    resources = ke.scene.SceneResourceManager(scene)
    assert resources.texture(ke.scene.InvalidResourceHandle) is None
    empty_texture = resources.register_texture("Empty", None)
    assert resources.texture(empty_texture) is None
    parent = scene.define_prim(path="/World", type=ke.scene.PrimType.XFORM)
    child = scene.define_prim(path="/World/Triangle", type=ke.scene.PrimType.MESH)
    parent.set_local_translation(ke.Vec3(3, 4, 5))
    parent.set_local_rotation_axis_angle(ke.Vec3(0, 0, 1), np.pi / 2)
    child.set_local_translation(ke.Vec3(1, 0, 0))
    np.testing.assert_allclose(child.get_world_translation(), [3, 5, 5], atol=1e-6)
    child.set_world_translation(ke.Vec3(3, 6, 5))
    np.testing.assert_allclose(child.get_local_translation(), [2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(
        child.compute_world_matrix(),
        np.asarray(parent.compute_world_matrix()) @ np.asarray(child.compute_local_matrix()),
        atol=1e-6,
    )

    mesh = ke.scene.MeshData.from_arrays(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], indices=[0, 1, 2]
    )
    handle = resources.register_mesh("Triangle", mesh, "memory://triangle")
    child.set_mesh_data(mesh)
    component = child.get_mesh_component()
    component.resource_handle = handle
    assert child.get_parent() is parent
    paths = []
    parent.traverse(lambda prim: paths.append(prim.get_path()))
    assert paths == ["/World", "/World/Triangle"]
    assert scene.get_prim_at_path("/World/Triangle") is child
    assert resources.mesh(handle) is child.resolve_mesh_data()
    assert len(scene.load_mesh("/World/Triangle").vertices) == 3
    token = ke.scene.Token("test:label")
    child.set_attribute_string(token.str(), "triangle")
    assert child.get_attribute_string(ke.scene.Token("test:label").str()) == "triangle"
    assert token.id() == ke.scene.Token("test:label").id()
    assert token.str() == "test:label"

    render = child.add_render_component()
    assert render.alpha_mode == ke.render.AlphaMode.OPAQUE
    render.transform_source = ke.render.TransformSource.SCENE_GRAPH
    render.double_sided = True
    assert render.double_sided
    shader = ke.scene.ShaderSourceResource()
    shader.stage = ke.render.ShaderType.VERTEX
    assert shader.stage == ke.render.ShaderType.VERTEX

    del mesh
    gc.collect()
    assert len(resources.mesh(handle).vertices) == 3
    assert scene.remove_prim("/World")
    assert not component.attached and component.owner is None
    assert not render.attached and render.owner is None
    assert len(resources.mesh(handle).vertices) == 3
    resources.clear()
    assert len(resources) == 0


def usd_scene():
    if not ke.scene.has_usd_support():
        assert not hasattr(ke.scene, "USDScene")
        try:
            ke.scene.create_backend(type=ke.scene.BackendType.USD)
        except RuntimeError as error:
            assert "USD support not compiled" in str(error)
        else:
            raise AssertionError("USD-disabled factory accepted USD")
        return False

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scene.usda"
        path.write_text('''#usda 1.0
def Xform "World" {
    double3 xformOp:translate = (3, 4, 5)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Mesh "Triangle" {
        double3 xformOp:translate = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }
}
''')
        scene = ke.scene.create_backend(type=ke.scene.BackendType.USD)
        assert isinstance(scene, ke.scene.USDScene)
        assert scene.load_scene(str(path))
        assert scene.list_meshes() == ["/World/Triangle"]
        mesh = scene.load_mesh("/World/Triangle")
        np.testing.assert_allclose(
            np.asarray(mesh.vertices),
            [[4, 4, 5], [5, 4, 5], [4, 5, 5]],
        )
        scene.create_xform("/Authored")
        scene.create_cube("/Authored/Cube", size=2)
        output = Path(directory) / "roundtrip.usdc"
        assert scene.save_scene(str(output))
        restored = ke.scene.USDScene()
        assert restored.load_scene(str(output))
        assert restored.list_meshes() == ["/World/Triangle"]
        np.testing.assert_allclose(
            np.asarray(restored.load_mesh("/World/Triangle").vertices),
            np.asarray(mesh.vertices),
        )
        authored = Path(directory) / "authored.usda"
        assert restored.save_scene(str(authored))
        assert 'def Cube "Cube"' in authored.read_text()
    return True


def main():
    native_scene()
    has_usd = usd_scene()
    assert_no_viewer_or_physics()
    print(f"PASS: scene data hierarchy/resources; USD {'roundtrip' if has_usd else 'disabled'}")


if __name__ == "__main__":
    main()
