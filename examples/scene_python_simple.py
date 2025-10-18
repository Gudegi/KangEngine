#!/usr/bin/env python3
"""
Simple USD Scene Example - File-based workflow
Uses USD files as the bridge between Python USD and KangEngine
"""


import sys
import os
sys.path.append(os.getcwd() + "/build")
import kangengine as ke

try:
    pass
    #import kangengine as ke
except ImportError:
    print("Error: kangengine module not found.")
    print("Build with: cmake --preset=vcpkg -DUSE_USD=ON -DIS_PYTHON_LIB=ON")
    print("Then: cmake --build build --target install")
    sys.exit(1)

try:
    from pxr import Usd, UsdGeom, Sdf, Gf
except ImportError:
    print("Error: pxr (OpenUSD) module not found.")
    print("Install with: pip install usd-core")
    sys.exit(1)


def main():
    print("=== KangEngine + Python OpenUSD Integration (File-based) ===\n")

    # Check USD support
    if not ke.scene.has_usd_support():
        print("Error: KangEngine was not built with USD support")
        return

    # ===========================================================
    # Part 1: Create scene with Python USD
    # ===========================================================
    print("1. Creating USD scene with Python OpenUSD API...")

    # Create USD Stage in Python
    stage = Usd.Stage.CreateInMemory()

    # Define scene structure
    world = UsdGeom.Xform.Define(stage, "/World")

    # Create some geometry with Python USD
    sphere1 = UsdGeom.Sphere.Define(stage, "/World/PythonSphere1")
    sphere1.GetRadiusAttr().Set(2.0)
    sphere1.AddTranslateOp().Set(Gf.Vec3d(-3, 0, 0))

    sphere2 = UsdGeom.Sphere.Define(stage, "/World/PythonSphere2")
    sphere2.GetRadiusAttr().Set(1.5)
    sphere2.AddTranslateOp().Set(Gf.Vec3d(3, 0, 0))

    # Create a custom mesh
    mesh = UsdGeom.Mesh.Define(stage, "/World/CustomTriangle")
    mesh.GetPointsAttr().Set([
        Gf.Vec3f(0, 0, 0),
        Gf.Vec3f(1, 0, 0),
        Gf.Vec3f(0.5, 1, 0),
    ])
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    mesh.AddTranslateOp().Set(Gf.Vec3d(0, 2, 0))

    # Create a procedural scene
    print("   Creating 10 procedural spheres...")
    for i in range(10):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/ProceduralSpheres/Sphere_{i}")
        sphere.GetRadiusAttr().Set(0.5)
        sphere.AddTranslateOp().Set(Gf.Vec3d(i * 1.5 - 7, 0, -5))

    # Set default prim
    stage.SetDefaultPrim(world.GetPrim())

    # Save to file
    output_file = "python_created_scene.usda"
    stage.Export(output_file)
    print(f"   Saved to: {output_file}\n")

    # ===========================================================
    # Part 2: Load into KangEngine
    # ===========================================================
    print("2. Loading scene into KangEngine...")

    # Create KangEngine USD scene
    ke_scene = ke.scene.USDScene()

    # Load the file we created
    if ke_scene.load_scene(output_file):
        print("   Scene loaded successfully!")

        # Print hierarchy
        print("\n3. Scene Hierarchy in KangEngine:")
        ke_scene.print_hierarchy()

        # List all meshes
        print("\n4. Meshes detected:")
        meshes = ke_scene.list_meshes()
        for mesh_path in meshes:
            print(f"   - {mesh_path}")

            # Get mesh data for rendering
            mesh_data = ke_scene.load_mesh(mesh_path)
            print(f"     (vertices: {len(mesh_data.vertices)}, indices: {len(mesh_data.indices)})")
    else:
        print("   Failed to load scene")
        return

    # ===========================================================
    # Part 3: Modify and Re-export
    # ===========================================================
    print("\n5. Modifying scene with Python USD...")

    # Re-open the stage to modify
    stage = Usd.Stage.Open(output_file)

    # Add more objects
    cube = UsdGeom.Cube.Define(stage, "/World/AddedCube")
    cube.GetSizeAttr().Set(1.5)
    cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 5))

    # Save modified scene
    modified_file = "python_modified_scene.usda"
    stage.Export(modified_file)
    print(f"   Saved modified scene to: {modified_file}")

    # Reload in KangEngine
    print("\n6. Reloading modified scene...")
    ke_scene_mod = ke.scene.USDScene()
    if ke_scene_mod.load_scene(modified_file):
        print("   Modified scene loaded!")
        meshes = ke_scene_mod.list_meshes()
        print(f"   Now has {len(meshes)} meshes")

    # ===========================================================
    # Part 4: KangEngine Creation
    # ===========================================================
    print("\n7. Creating scene directly in KangEngine...")

    # Create new scene
    ke_scene_new = ke.scene.USDScene()
    ke_scene_new.create_new()

    # Use KangEngine convenience API
    ke_scene_new.create_xform("/EngineWorld")
    ke_scene_new.create_sphere("/EngineWorld/EngineSphere", radius=3.0)
    ke_scene_new.create_cube("/EngineWorld/EngineCube", size=2.0)

    # Save
    engine_file = "kangengine_created_scene.usda"
    ke_scene_new.save_scene(engine_file)
    print(f"   Saved KangEngine scene to: {engine_file}")

    # Now open with Python USD to verify
    print("\n8. Verifying with Python USD...")
    verify_stage = Usd.Stage.Open(engine_file)
    print(f"   Stage root layer: {verify_stage.GetRootLayer().identifier}")
    print(f"   Default prim: {verify_stage.GetDefaultPrim().GetPath()}")

    for prim in verify_stage.Traverse():
        print(f"   - {prim.GetPath()} [{prim.GetTypeName()}]")

    # ===========================================================
    # Summary
    # ===========================================================
    print("\n=== Workflow Summary ===")
    print("✓ Created scene with Python OpenUSD")
    print("✓ Loaded into KangEngine for rendering")
    print("✓ Modified scene with Python USD")
    print("✓ Created scene with KangEngine API")
    print("✓ Verified with Python USD")
    print("\nThis file-based workflow allows:")
    print("  • Python USD for scene creation/editing")
    print("  • KangEngine for rendering")
    print("  • Full USD feature set (layers, variants, etc.)")
    print("\nFiles created:")
    print(f"  • {output_file}")
    print(f"  • {modified_file}")
    print(f"  • {engine_file}")
    print("\nView with: usdview <filename>.usda")


if __name__ == "__main__":
    main()
