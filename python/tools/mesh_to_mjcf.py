"""Export an OBJ as one MJCF rigid body with convex collision parts."""

from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import kangengine as ke

import convex_decomposition as decomposition


def _numbers(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _write_obj(path: Path, vertices, faces) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for vertex in vertices:
            stream.write(f"v {_numbers(vertex)}\n")
        for face in faces:
            stream.write("f " + " ".join(str(int(i) + 1) for i in face) + "\n")


def mesh_to_mjcf(
    obj_path: str | Path,
    output_dir: str | Path,
    *,
    mass: float | None = None,
    result_path: str | Path | None = None,
    rebuild: bool = False,
) -> Path:
    """Save visual geometry, convex parts, and a free-body MJCF.

    Input coordinates are kept unchanged and interpreted as meters, with Y up.
    Materials and textures are not exported. Mass properties use PhysX's cooked
    compound at density 1 kg/m³. If mass is given in kg, scale the inertia to
    match it while preserving the center of mass and principal axes.
    """
    if mass is not None and (not isfinite(mass) or mass <= 0):
        raise ValueError("mass must be finite and greater than zero (kg)")
    obj_path = Path(obj_path)
    output_dir = Path(output_dir)
    mesh: ke.scene.MeshData = ke.asset.load_obj(path=str(obj_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
    parts, metadata = decomposition.decompose_mesh(
        vertices=vertices,
        faces=faces,
        options={"threshold": 0.05, "seed": 0},
        result_path=result_path if result_path is not None else output_dir / "coacd.npz",
        rebuild=rebuild,
    )

    # Author local mesh coordinates once; all geoms share the body's frame.
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco", model=obj_path.stem)
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes")
    ET.SubElement(root, "option", gravity="0 -9.81 0")
    defaults = ET.SubElement(root, "default")
    collision = ET.SubElement(defaults, "default", {"class": "collision"})
    ET.SubElement(collision, "geom", type="mesh", group="3")
    assets = ET.SubElement(root, "asset")
    _write_obj(mesh_dir / "visual.obj", vertices, faces)
    ET.SubElement(assets, "mesh", name="visual", file="visual.obj")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name=obj_path.stem)
    ET.SubElement(body, "joint", name="root", type="free")
    ET.SubElement(
        body, "geom", name="visual", type="mesh", mesh="visual",
        contype="0", conaffinity="0", group="1", density="0", rgba="0.8 0.8 0.8 1",
    )
    for i, part in enumerate(parts):
        name = f"part_{i:03d}"
        _write_obj(mesh_dir / f"{name}.obj", part.vertices, part.faces)
        ET.SubElement(assets, "mesh", name=name, file=f"{name}.obj")
        ET.SubElement(body, "geom", {"name": name, "class": "collision", "mesh": name})

    # Reuse the same cooking and mass calculation as the runtime collider.
    world = ke.physics.PhysicsWorld(ke.physics.PhysicsConfig.y_up())
    actor = world.create_dynamic_convex_compound(
        [ke.physics.ConvexMeshPart(p.vertices, p.faces) for p in parts],
        pos=(0.0, 0.0, 0.0), density=1.0,
    )
    try:
        computed_mass = actor.get_mass()
        output_mass = computed_mass if mass is None else mass
        # Uniform density changes scale inertia without moving the COM.
        inertia = actor.get_mass_space_inertia() * (output_mass / computed_mass)
        pose = actor.get_local_com_pose()
        ET.SubElement(
            body, "inertial", pos=_numbers(pose[:3]),
            quat=_numbers(pose[[6, 3, 4, 5]]), mass=f"{output_mass:.9g}",
            diaginertia=_numbers(inertia),
        )
    finally:
        actor.release()

    destination = output_dir / f"{obj_path.stem}.xml"
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    print(f"Saved {destination}: {len(parts)} parts, cache_hit={metadata['cache_hit']}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mass", type=float, help="Total mass in kg; otherwise use density 1 kg/m³.")
    parser.add_argument("--result", type=Path, help="Reuse an existing decomposition NPZ.")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    mesh_to_mjcf(
        args.obj_path, args.output_dir, mass=args.mass,
        result_path=args.result, rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
