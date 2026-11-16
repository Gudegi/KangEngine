"""Check PLY attributes and invalid inputs."""

import struct

import numpy as np
import pytest

import kangengine as ke


def write_ply(tmp_path, properties, payload, *, count=3, extra="", encoding="ascii"):
    path = tmp_path / "sample.ply"
    header = (
        f"ply\nformat {encoding} 1.0\ncomment test fixture\n"
        f"element vertex {count}\n{properties}{extra}end_header\n"
    )
    path.write_bytes(header.encode("ascii") + payload)
    return str(path)


XYZ = "property float x\nproperty float y\nproperty float z\n"
RGB = "property uchar red\nproperty uchar green\nproperty uchar blue\n"
FACE = "element face 1\nproperty list uchar int vertex_indices\n"


def test_ascii_mesh_preserves_attributes_and_polygon_winding(tmp_path):
    properties = (
        "property uchar blue\nproperty float z\nproperty float x\n"
        "property uchar red\nproperty float y\nproperty uchar green\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float texture_u\nproperty float texture_v\n"
    )
    path = write_ply(tmp_path, properties, (
        "0 0 0 255 0 0 0 0 1 0 0\n"
        "0 0 1 0 0 255 0 0 1 1 0\n"
        "255 0 1 0 1 0 0 0 1 1 1\n"
        "255 0 0 255 1 255 0 0 1 0 1\n"
        "4 0 1 2 3\n"
    ).encode(), count=4, extra=FACE)
    mesh = ke.asset.load_ply(path=path)
    np.testing.assert_array_equal(mesh.vertices, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    np.testing.assert_array_equal(mesh.indices, [0, 1, 2, 0, 2, 3])
    np.testing.assert_array_equal(mesh.colors, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    np.testing.assert_array_equal(mesh.normals, [[0, 0, 1]] * 4)
    np.testing.assert_array_equal(mesh.uvs, [[0, 0], [1, 0], [1, 1], [0, 1]])


@pytest.mark.parametrize("endian,encoding", [
    ("<", "binary_little_endian"), (">", "binary_big_endian"),
])
def test_binary_mesh_skips_unknown_properties_and_elements(tmp_path, endian, encoding):
    properties = XYZ.replace("float", "float64") + RGB + (
        "property list uint8 int16 neighbors\nproperty float32 confidence\n"
    )
    payload = b"".join(
        struct.pack(endian + "dddBBBBhhf", *point, *color, 2, -3, 9, 0.5)
        for point, color in zip(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        )
    )
    payload += struct.pack(endian + "HI", 1, 42)
    payload += struct.pack(endian + "Biii", 3, 0, 1, 2)
    extra = "element edge 1\nproperty list ushort uint vertices\n" + FACE
    path = write_ply(tmp_path, properties, payload, extra=extra, encoding=encoding)
    mesh = ke.asset.load_ply(path)
    np.testing.assert_array_equal(mesh.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(mesh.colors, np.eye(3))
    np.testing.assert_array_equal(mesh.indices, [0, 1, 2])
    np.testing.assert_allclose(mesh.normals, [[0, 0, 1]] * 3)
    np.testing.assert_array_equal(mesh.uvs, np.zeros((3, 2)))


def test_points_keep_missing_attributes_empty(tmp_path):
    path = write_ply(tmp_path, XYZ, b"+1 2e0 -3", count=1)
    mesh = ke.asset.load_ply(path)
    np.testing.assert_array_equal(mesh.vertices, [[1, 2, -3]])
    assert not mesh.indices and not mesh.normals and not mesh.uvs and not mesh.colors


@pytest.mark.parametrize("color_type,values,expected", [
    ("uchar", "255 128 0", [1, 128 / 255, 0]),
    ("ushort", "65535 32768 0", [1, 32768 / 65535, 0]),
    ("float", "1 0.5 0.02", [1, 0.5, 0.02]),
])
def test_color_normalization_and_explicit_srgb(tmp_path, color_type, values, expected):
    path = write_ply(tmp_path, XYZ + RGB.replace("uchar", color_type),
                     f"0 0 0 {values}\n".encode(), count=1)
    np.testing.assert_allclose(ke.asset.load_ply(path).colors, [expected], rtol=1e-6)
    rgb = np.asarray(expected)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    np.testing.assert_allclose(ke.asset.load_ply(path, colors_srgb=True).colors,
                               [linear], rtol=1e-6)


def test_faces_before_vertices_and_crlf_header(tmp_path):
    path = tmp_path / "face_first.ply"
    path.write_bytes((
        "ply\r\nformat ascii 1.0\r\n" + FACE.replace("vertex_indices", "vertex_index")
        + "element vertex 3\n" + XYZ + "end_header\n"
        + "3 0 1 2\n0 0 0\n1 0 0\n0 1 0\n"
    ).encode())
    np.testing.assert_array_equal(ke.asset.load_ply(str(path)).indices, [0, 1, 2])


@pytest.mark.parametrize("properties,payload,count,extra", [
    (XYZ, b"0 0 nan\n", 1, ""),
    (XYZ, b"0 0 0oops\n", 1, ""),
    (XYZ + "property uchar red\n", b"0 0 0 255\n", 1, ""),
    (XYZ.replace("property float z\n", ""), b"0 0\n", 1, ""),
    (XYZ, b"0 0 0\n", 1000000000, ""),
    (XYZ, b"0 0 0\n3 0 1 2\n", 1, FACE),
    (XYZ, b"0 0 0\n3 0 -1 0\n", 1, FACE),
    (XYZ, b"0 0 0\n2 0 0\n", 1, FACE),
    (XYZ, b"0 0 0\n-1\n", 1, FACE.replace("uchar", "char")),
    (XYZ, b"0 0 0\n3 0 0 0\n", 1, FACE.replace("int vertex", "float vertex")),
    (XYZ, b"0 0 0\n0\n", 1, "element face 1\nproperty int ignored\n"),
    (XYZ + RGB.replace("uchar", "float"), b"0 0 0 1 2 0\n", 1, ""),
])
def test_invalid_payload_reports_file(tmp_path, properties, payload, count, extra):
    path = write_ply(tmp_path, properties, payload, count=count, extra=extra)
    with pytest.raises(RuntimeError, match="PLY .*sample.ply"):
        ke.asset.load_ply(path)


def test_truncated_binary_list(tmp_path):
    payload = struct.pack("<fffBii", 0, 0, 0, 3, 0, 0)
    path = write_ply(tmp_path, XYZ, payload, count=1, extra=FACE,
                     encoding="binary_little_endian")
    with pytest.raises(RuntimeError, match="truncated list"):
        ke.asset.load_ply(path)


@pytest.mark.parametrize("header", [
    b"not_ply\n", b"ply\nformat ascii 2.0\nend_header\n",
    b"ply\nformat ascii 1.0\n", b"ply\nend_header\n",
    b"ply\nformat ascii 1.0\nelement vertex -1\nend_header\n",
])
def test_invalid_header(tmp_path, header):
    path = tmp_path / "broken.ply"
    path.write_bytes(header)
    with pytest.raises(RuntimeError, match="PLY .*broken.ply"):
        ke.asset.load_ply(str(path))


def test_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="cannot open file"):
        ke.asset.load_ply(str(tmp_path / "missing.ply"))


def write_indexed_uv_ply(tmp_path, encoding="ascii"):
    header = (
        f"ply\nformat {encoding} 1.0\nelement vertex 4\n" + XYZ
        + "property float nx\nproperty float ny\nproperty float nz\n" + RGB
        + "element face 2\nproperty list uchar int vertex_indices\n"
        + "element multi_texture_vertex 5\nproperty uchar tx\nproperty float u\nproperty float v\n"
        + "element multi_texture_face 2\nproperty uchar tx\nproperty uint tn\n"
        + "property list uchar int texture_vertex_indices\nend_header\n"
    )
    records = [
        ("ffffffBBB", (0, 0, 0, 0, 0, 1, 255, 0, 0)),
        ("ffffffBBB", (1, 0, 0, 0, 0, 1, 0, 255, 0)),
        ("ffffffBBB", (1, 1, 0, 0, 0, 1, 0, 0, 255)),
        ("ffffffBBB", (0, 1, 0, 0, 0, 1, 255, 255, 255)),
        ("Biii", (3, 0, 1, 2)), ("Biii", (3, 0, 2, 3)),
        ("Bff", (0, 0, 0)), ("Bff", (0, 1, 0)), ("Bff", (0, 1, 1)),
        ("Bff", (0, 0.5, 0)), ("Bff", (0, 0, 1)),
        # tn identifies source faces, not record order.
        ("BIBiii", (0, 1, 3, 3, 2, 4)), ("BIBiii", (0, 0, 3, 0, 1, 2)),
    ]
    if encoding == "ascii":
        payload = "".join(" ".join(map(str, values)) + "\n" for _, values in records).encode()
    else:
        endian = "<" if encoding == "binary_little_endian" else ">"
        payload = b"".join(struct.pack(endian + fmt, *values) for fmt, values in records)
    path = tmp_path / "indexed_uv.ply"
    path.write_bytes(header.encode() + payload)
    return path


@pytest.mark.parametrize("encoding", ["ascii", "binary_little_endian", "binary_big_endian"])
def test_uv_seams_preserve_geometry_normals_and_colors(tmp_path, encoding):
    mesh = ke.asset.load_ply(str(write_indexed_uv_ply(tmp_path, encoding)))
    ids = np.asarray(mesh.indices)
    assert len(mesh.vertices) == 5, "only the vertex at the UV seam should split"
    np.testing.assert_array_equal(np.asarray(mesh.vertices)[ids],
                                  [[0, 0, 0], [1, 0, 0], [1, 1, 0],
                                   [0, 0, 0], [1, 1, 0], [0, 1, 0]])
    np.testing.assert_array_equal(np.asarray(mesh.uvs)[ids],
                                  [[0, 1], [1, 1], [1, 0], [0.5, 1], [1, 0], [0, 0]])
    np.testing.assert_array_equal(np.asarray(mesh.colors)[ids],
                                  [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                   [1, 0, 0], [0, 0, 1], [1, 1, 1]])
    np.testing.assert_array_equal(mesh.normals, [[0, 0, 1]] * 5)


@pytest.mark.parametrize("old,new,message", [
    ("0 1 3 3 2 4", "0 1 3 3 2 9", "index out of range"),
    ("0 1 3 3 2 4", "0 2 3 3 2 4", "tn must identify"),
    ("0 1 3 3 2 4", "0 0 3 3 2 4", "tn must identify"),
    ("0 1 3 3 2 4", "0 1 4 3 2 4 0", "corner counts differ"),
    ("0 0 3 0 1 2", "1 0 3 0 1 2", "multiple textures"),
    ("0 0.5 0", "1 0.5 0", "identifiers differ"),
    ("0 0.5 0", "0 nan 0", "finite float"),
    ("property float u", "property float unused", "incomplete texture element"),
    ("property uchar tx", "property float tx", "integer tx/tn"),
    ("element multi_texture_face 2", "element multi_texture_face 1", "cover each face"),
])
def test_invalid_indexed_uv_mapping(tmp_path, old, new, message):
    path = write_indexed_uv_ply(tmp_path)
    path.write_text(path.read_text().replace(old, new), encoding="ascii")
    with pytest.raises(RuntimeError, match=message):
        ke.asset.load_ply(str(path))


def test_empty_texture_tables_preserve_points(tmp_path):
    extra = (
        "element face 0\nproperty list uchar int vertex_indices\n"
        "element multi_texture_vertex 0\nproperty uchar tx\nproperty float u\nproperty float v\n"
        "element multi_texture_face 0\nproperty uchar tx\nproperty uint tn\n"
        "property list uchar int texture_vertex_indices\n"
    )
    path = write_ply(tmp_path, XYZ, b"1 2 3\n", count=1, extra=extra)
    mesh = ke.asset.load_ply(path)
    np.testing.assert_array_equal(mesh.vertices, [[1, 2, 3]])
    assert not mesh.indices and not mesh.uvs
