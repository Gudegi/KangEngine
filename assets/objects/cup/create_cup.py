"""Create the hollow ceramic cup used by the convex decomposition example.

Run with Blender: blender --background --factory-startup --python <this file>.
The OBJ uses Y up and meters; the cup is deliberately oversized for the demo.
"""

from itertools import pairwise
from math import cos, pi, sin
from pathlib import Path

import bmesh
import bpy

OUTPUT = Path(__file__).resolve().parent
PROFILE = [
    (0.0, 0.0), (0.72, 0.0), (0.82, 0.04), (0.86, 0.10),
    (0.88, 0.22), (0.90, 1.50), (0.895, 1.60), (0.875, 1.65),
    (0.845, 1.67), (0.795, 1.67), (0.765, 1.65), (0.745, 1.60),
    (0.75, 0.23), (0.72, 0.18), (0.64, 0.16), (0.0, 0.16),
]


def create_mesh(name, vertices, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_body():
    vertices, faces, rings = [], [], []
    segments = 64
    for radius, height in PROFILE:
        ring = []
        for i in range(segments if radius else 1):
            angle = 2 * pi * i / segments
            ring.append(len(vertices))
            vertices.append((radius * cos(angle), radius * sin(angle), height))
        rings.append(ring)
    for lower, upper in pairwise(rings):
        for i in range(segments):
            j = (i + 1) % segments
            if len(lower) == 1:
                faces.append((lower[0], upper[j], upper[i]))
            elif len(upper) == 1:
                faces.append((lower[i], lower[j], upper[0]))
            else:
                faces.append((lower[i], lower[j], upper[j], upper[i]))
    return create_mesh("Cup", vertices, faces)


def create_handle():
    vertices, faces = [], []
    segments, sides = 32, 12
    for i in range(segments + 1):
        angle = -pi / 2 + pi * i / segments
        # The half ellipse meets the outer wall without entering the cavity.
        nx, nz = 0.52 * cos(angle), 0.65 * sin(angle)
        length = (nx * nx + nz * nz) ** 0.5
        for j in range(sides):
            around = 2 * pi * j / sides
            vertices.append((
                0.83 + 0.65 * cos(angle) + 0.12 * cos(around) * nx / length,
                0.12 * sin(around),
                0.85 + 0.52 * sin(angle) + 0.12 * cos(around) * nz / length,
            ))
    for i in range(segments):
        for j in range(sides):
            k = (j + 1) % sides
            faces.append((i * sides + j, i * sides + k,
                          (i + 1) * sides + k, (i + 1) * sides + j))
    faces.extend([tuple(reversed(range(sides))),
                  tuple(segments * sides + j for j in range(sides))])
    return create_mesh("Handle", vertices, faces)


def main():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    cup, handle = create_body(), create_handle()
    bpy.context.view_layer.objects.active = cup
    cup.select_set(True)

    # Fuse the handle into the wall so decomposition receives one solid mesh.
    union = cup.modifiers.new("Join handle", "BOOLEAN")
    union.operation = "UNION"
    union.solver = "EXACT"
    union.object = handle
    bpy.ops.object.modifier_apply(modifier=union.name)
    bpy.data.objects.remove(handle, do_unlink=True)
    bm = bmesh.new()
    bm.from_mesh(cup.data)
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    assert all(edge.is_manifold for edge in bm.edges), "Cup must be a closed solid"
    assert bm.calc_volume(signed=True) > 0, "Cup normals must face outward"
    bm.to_mesh(cup.data)
    bm.free()
    for polygon in cup.data.polygons:
        polygon.use_smooth = True

    material = bpy.data.materials.new("Ceramic")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.72, 0.88, 0.78, 1.0)
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = 0.2
    cup.data.materials.append(material)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT / "cup.blend"))
    bpy.ops.wm.obj_export(
        filepath=str(OUTPUT / "cup.obj"), export_selected_objects=True,
        forward_axis="NEGATIVE_Z", up_axis="Y", export_materials=False,
        export_triangulated_mesh=True,
    )
    print(f"CUP_SAVED vertices={len(cup.data.vertices)} triangles={len(cup.data.polygons)}")


if __name__ == "__main__":
    main()
