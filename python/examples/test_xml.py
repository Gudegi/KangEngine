"""
Test XML Robot Viewer - Python version of test_xml.cpp
Loads Unitree H1 robot from MJCF, displays with Phong lighting.

Note: ImGui UI is not available in Python bindings, so the
slider/scene tree panels from the C++ version are omitted.
"""

import sys
import os
import math
import numpy as np

import kangengine as ke
from kangengine import scene, animation

from pathlib import Path
current_file = Path(__file__).resolve()


project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
try:
    from references.isaacsim.poselib.poselib.core.my_rotation3d import *
    from references.isaacsim.poselib.poselib.skeleton.my_skeleton3d import SkeletonState, SkeletonMotion, SkeletonTree
except ImportError as e:
    print(f"Import Failed: {e}")
    print(f"Search Path was: {project_root}")

# --- GLSL Shaders (same as test_xml.cpp) ---

stl_vs = """
#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
out vec3 Normal;
out vec3 FragPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMat;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    FragPos = vec3(view * model * vec4(aPos, 1.0f));
    Normal = normalMat * aNormal;
}
"""

stl_fs = """
#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
uniform vec4 objColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 camPos;
void main() {
    float ambientStrength = 0.15f;
    vec3 ambient = ambientStrength * lightColor;
    vec3 NormalDir = normalize(Normal);
    vec3 lightToFaceDir = normalize(FragPos - lightPos);
    float diff = max(dot(-lightToFaceDir, NormalDir), 0.0);
    vec3 diffuse = diff * lightColor;
    float specularStrength = 0.5;
    int shininess = 32;
    vec3 viewDir = normalize(FragPos - camPos);
    vec3 reflectDir = reflect(lightToFaceDir, NormalDir);
    float spec = pow(max(dot(-viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;
    vec3 phong = ambient + diffuse + specular;
    FragColor = vec4(objColor) * vec4(phong, 1.0f);
}
"""

test_vs = """
#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMat;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
    FragPos = vec3(view * model * vec4(aPos, 1.0f));
    Normal = normalMat * aNormal;
}
"""

light_fs = """
#version 410 core
out vec4 FragColor;
in vec2 TexCoord;
uniform vec3 lightColor;
void main() {
   FragColor = vec4(lightColor, 1.0);
}
"""

checker_fs = """
#version 410 core
out vec4 FragColor;
in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;
uniform vec4 checkerColor1;
uniform vec4 checkerColor2;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 camPos;
float checkerboardPattern(vec2 uv) {
    return mod((floor(uv.x * 10.f) + floor(uv.y * 10.f)), 2.f);
}
void main() {
    float checkerType = checkerboardPattern(TexCoord);
    vec4 objColor = checkerColor1 * (1 - checkerType) + checkerColor2 * checkerType;
    float ambientStrength = 0.15f;
    vec3 ambient = ambientStrength * lightColor;
    vec3 NormalDir = normalize(Normal);
    vec3 lightToFaceDir = normalize(FragPos - lightPos);
    float diff = max(dot(-lightToFaceDir, NormalDir), 0.0);
    vec3 diffuse = diff * lightColor;
    float specularStrength = 0.5;
    int shininess = 32;
    vec3 viewDir = normalize(FragPos - camPos);
    vec3 reflectDir = reflect(lightToFaceDir, NormalDir);
    float spec = pow(max(dot(-viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;
    vec3 phong = ambient + diffuse + specular;
    FragColor = vec4(objColor) * vec4(phong, 1.0f);
}
"""


def transform_point(mat, point):
    """Compute mat4 * vec4(point, 1.0) -> vec3 (view-space transform)"""
    x, y, z = point.x, point.y, point.z
    col0, col1, col2, col3 = mat[0], mat[1], mat[2], mat[3]
    rx = col0.x * x + col1.x * y + col2.x * z + col3.x
    ry = col0.y * x + col1.y * y + col2.y * z + col3.y
    rz = col0.z * x + col1.z * y + col2.z * z + col3.z
    return ke.vec3(rx, ry, rz)


motion_file = "/Users/asaid/Dev/sample_isaac/assets/IsaacSim_motions/unitree_h1/samples/SwingDancing1.npy"
# motion_file = "/Users/asaid/Dev/sample_isaac/assets/IsaacSim_motions/unitree_h1/samples/StandardRun.npy"
asset_dir = os.path.join(os.path.dirname(ke.__file__), 'assets')
mjcf_path = os.path.join(asset_dir, 'external', 'retargetted','unitree_h1', 'unitree_h1.xml')
motion : SkeletonMotion = SkeletonMotion.from_file(motion_file)
frame_idx = 0
root_translation = motion.global_translation[:, 0].numpy()
root_translation[:, 2] += 0.3 
local_rotation = motion.local_rotation.numpy()
global_rotation = motion.global_rotation.numpy()

class RobotViewer(ke.App):
    def setup(self):
        self.setRenderHz(1. / 30.)
        device = self.getGraphicsDevice()

        # Create shaders
        self.stl_shader = device.createShader(stl_vs, stl_fs)
        self.light_shader = device.createShader(test_vs, light_fs)
        self.plane_shader = device.createShader(test_vs, checker_fs)

        self.light_pos = ke.vec3(0.5, -2.0, 3.0)

        # Set initial shader uniforms
        self.stl_shader.use()
        self.stl_shader.setVec3("lightColor", 1.0, 1.0, 1.0)

        self.light_shader.use()
        self.light_shader.setVec3("lightColor", 1.0, 1.0, 1.0)

        self.plane_shader.use()
        self.plane_shader.setVec4("checkerColor1",
                                  ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.plane_shader.setVec4("checkerColor2",
                                  ke.vec4(0.77, 0.93, 0.78, 1.0))
        self.plane_shader.setMat4("model", ke.mat4(1.0))

        # Light sphere
        sphere_data = scene.Prim.create_sphere_data(1.0, 12, 11)
        light_prim = self.getScene().define_prim("/light",
                                                 scene.PrimType.Mesh)
        light_prim.set_mesh_data(sphere_data)
        self.addShapePrim(self.light_shader, light_prim)
        light_prim.set_attribute_vec3("xformOp:translate", self.light_pos)
        light_prim.set_attribute_vec3("xformOp:scale",
                                      ke.vec3(0.05, 0.05, 0.05))

        # Ground plane (rotated for Z-up)
        ground_prim = self.getScene().define_prim("/ground",
                                                  scene.PrimType.Mesh)
        plane_data = scene.Prim.create_plane_data(10.0)
        ground_prim.set_mesh_data(plane_data)
        # upAxisRotation(Z): 90 degrees around X axis
        cos45 = math.cos(math.radians(45))
        sin45 = math.sin(math.radians(45))
        ground_prim.add_rotate_quaternion_op(
            ke.quat(cos45, sin45, 0.0, 0.0))
        ground_prim.set_attribute_vec3("xformOp:translate", ke.vec3(0, 0, 0))
        self.addShapePrim(self.plane_shader, ground_prim)

        # Load MJCF robot
        asset_dir = os.path.join(os.path.dirname(ke.__file__), 'assets')
        mjcf_path = os.path.join(
            asset_dir, 'external', 'retargetted',
            'unitree_h1', 'unitree_h1.xml')
        self.robot = animation.Robot.from_mjcf(
            mjcf_path, self.getScene(), self.stl_shader, self)

        print(f"Robot loaded: {self.robot.num_bodies()} bodies")
        self.checkError()

    def preRender(self):
        self.checkError()

    def render(self):
        global frame_idx
        view = self.getViewMatrix()
        cam_pos = ke.vec3(0.0, 0.0, 0.0)
        
        root_pos = root_translation[frame_idx]
        self.robot.set_root_translation(ke.vec3(float(root_pos[0]), float(root_pos[1]), float(root_pos[2])))
        for i in range(self.robot.num_bodies()):
            quat = local_rotation[frame_idx][i]
            self.robot.set_joint_rotation(i, ke.quat(quat[0], quat[1], quat[2], quat[3]))
        self.robot.apply_pose()
        frame_idx += 1
        
        if frame_idx >= root_translation.shape[0]:
            frame_idx = 0

        light_view = transform_point(view, self.light_pos)
        cam_view = transform_point(view, cam_pos)

        self.stl_shader.use()
        self.stl_shader.setVec3("lightColor", 1.0, 1.0, 1.0)
        self.stl_shader.setVec3("lightPos", light_view)
        self.stl_shader.setVec3("camPos", cam_view)

        self.plane_shader.use()
        self.plane_shader.setVec3("lightColor", 1.0, 1.0, 1.0)
        self.plane_shader.setVec3("lightPos", light_view)
        self.plane_shader.setVec3("camPos", cam_view)

        self.checkError()

    def postRender(self):
        pass


if __name__ == "__main__":
    app = RobotViewer()
    app.initialize(1920, 1080, False, ke.Z, ke.OpenGL)
    app.start()
