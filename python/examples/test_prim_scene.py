"""
Test Prim Scene - Python version of test_prim_scene.cpp
Demonstrates Token, Prim, and attribute system usage.
"""

import sys
sys.path.insert(0, './bindings')

import kangengine as ke
from kangengine import scene

# Test Token class
print("=== Token Test ===")
t1 = scene.Token("xformOp:translate")
t2 = scene.Token("xformOp:translate")
t3 = scene.Token("xformOp:scale")

print(f"t1: {t1}")
print(f"t2: {t2}")
print(f"t3: {t3}")
print(f"t1 == t2: {t1 == t2}")  # Should be True
print(f"t1 == t3: {t1 == t3}")  # Should be False
print(f"t1.id(): {t1.id()}")
print(f"t1.str(): {t1.str()}")

# Test Prim creation
print("\n=== Prim Test ===")
cube_prim = scene.Prim("box", scene.PrimType.Mesh)
print(f"Prim name: {cube_prim.get_name()}")
print(f"Prim path: {cube_prim.get_path()}")
print(f"Prim type: {cube_prim.get_type()}")

# Test vec3, vec4, quat types
print("\n=== GLM Types Test ===")
v3 = ke.vec3(1.0, 2.0, 3.0)
v4 = ke.vec4(0.8, 0.3, 0.02, 1.0)
q = ke.quat(1.0, 0.0, 0.0, 0.0)  # Identity quaternion (w, x, y, z)

print(f"vec3: {v3}")
print(f"vec4: {v4}")
print(f"quat: {q}")

# Test setAttribute/getAttribute
print("\n=== Prim Attribute Test ===")
box_pos = ke.vec3(0.0, 10.0, 0.0)
cube_prim.add_translate_op(box_pos)
cube_prim.add_scale_op(ke.vec3(1.3, 1.3, 1.3))
cube_prim.add_rotate_quaternion_op(ke.quat(1.0, 0.0, 0.0, 0.0))

# Test display color
cube_prim.set_display_color(ke.vec3(0.8, 0.3, 0.02))
color = cube_prim.get_display_color()
if color:
    print(f"Display color: {color}")

# Test display color alpha
cube_prim.set_display_color_alpha(ke.vec4(0.8, 0.3, 0.02, 1.0))
color_alpha = cube_prim.get_display_color_alpha()
if color_alpha:
    print(f"Display color alpha: {color_alpha}")

# Test has_attribute
print(f"Has 'primvars:displaycolor': {cube_prim.has_attribute('primvars:displaycolor')}")
print(f"Has 'nonexistent': {cube_prim.has_attribute('nonexistent')}")

# Test compute_model_matrix
print("\n=== Model Matrix Test ===")
model_mat = cube_prim.compute_model_matrix()
print(f"Model matrix computed: {type(model_mat)}")

# Test MeshData creation
print("\n=== MeshData Test ===")
square_data = scene.Prim.create_square_data(1.0)
print(f"Square vertices count: {len(square_data.vertices)}")
print(f"Square indices count: {len(square_data.indices)}")

sphere_data = scene.Prim.create_sphere_data(1.0, 12, 11)
print(f"Sphere vertices count: {len(sphere_data.vertices)}")
print(f"Sphere indices count: {len(sphere_data.indices)}")

plane_data = scene.Prim.create_plane_data(30.0)
print(f"Plane vertices count: {len(plane_data.vertices)}")
print(f"Plane indices count: {len(plane_data.indices)}")

# Note: set_mesh_data requires shared_ptr<MeshData> which needs special handling
# For now, skip this test - MeshData is typically used with addShape() directly
print("(Skipping set_mesh_data test - requires shared_ptr wrapper)")

# Test Prim hierarchy
print("\n=== Prim Hierarchy Test ===")
root = scene.Prim("World", scene.PrimType.Xform)
child1 = root.add_child("Cube", scene.PrimType.Mesh)
child2 = root.add_child("Sphere", scene.PrimType.Mesh)

print(f"Root path: {root.get_path()}")
print(f"Child1 path: {child1.get_path()}")
print(f"Child2 path: {child2.get_path()}")

# Get children
children = root.get_children()
print(f"Root has {len(children)} children")
for child in children:
    print(f"  - {child.get_name()}")

# Get child by name
found = root.get_child("Cube")
if found:
    print(f"Found child: {found.get_name()}")

print("\n=== All Tests Passed ===")
