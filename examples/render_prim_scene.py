"""
Render Prim Scene - Python version of test_prim_scene.cpp with rendering
Demonstrates full rendering pipeline with Prim and shader system.
"""

import sys
from pyglm import glm
sys.path.insert(0, './bindings')

import kangengine as ke
from kangengine import scene

# Shader sources (same as test_prim_scene.cpp)
LIGHT_FS = """
#version 410 core

out vec4 FragColor;

in vec2 TexCoord;
uniform vec3 lightColor;

void main() {
   FragColor = vec4(lightColor, 1.0);
}
"""

TEST_VS = """
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

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
    FragPos = vec3(view * model * vec4(aPos, 1.0f));
    Normal = normalMat * aNormal;
}
"""

TEST_FS = """
#version 410 core

out vec4 FragColor;

in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;

uniform vec4 objColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 camPos;

void main() {
    // ambient light
    float ambientStrength = 0.15f;
    vec3 ambient = ambientStrength * lightColor;

    // diffuse light
    vec3 NormalDir = normalize(Normal);
    vec3 lightToFaceDir = normalize(FragPos - lightPos);
    float diff = max(dot(-lightToFaceDir, NormalDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // specular light
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

CHECKER_FS = """
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

float checkerboardPattern(vec2 uv){
    // return 0 or 1, 10 x 10 boxes
    return mod((floor(uv.x * 10.f) + floor(uv.y * 10.f)), 2.f);
}

void main() {
    float checkerType = checkerboardPattern(TexCoord);
    vec4 objColor = checkerColor1 * (1 - checkerType) + checkerColor2 * checkerType;

    // ambient light
    float ambientStrength = 0.15f;
    vec3 ambient = ambientStrength * lightColor;    

    // diffuse light
    vec3 NormalDir = normalize(Normal);
    vec3 lightToFaceDir = normalize(FragPos - lightPos);
    float diff = max(dot(-lightToFaceDir, NormalDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // specular light
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

SCR_WIDTH = 1920
SCR_HEIGHT = 1080

class MyApp(ke.App):
    def __init__(self):
        super().__init__()
        self.cube_shader = None
        self.light_shader = None
        self.plane_shader = None
        self.cube_prim = None
        self.light_prim = None

        self.light_pos = ke.vec3(-5.0, 5.0, 1.0)
        self.box_pos = ke.vec3(0.0, 2.0, 0.0)
        self.size = 1.3
        self.color = [0.8, 0.3, 0.02, 1.0]

    def setup(self):
        device = self.getGraphicsDevice()

        # Create shaders
        self.cube_shader = device.createShader(TEST_VS, TEST_FS)
        self.light_shader = device.createShader(TEST_VS, LIGHT_FS)
        self.plane_shader = device.createShader(TEST_VS, CHECKER_FS)

        # Setup cube shader
        self.cube_shader.use()
        self.cube_shader.setColor("objColor", self.color[0], self.color[1], self.color[2], self.color[3])
        self.cube_shader.setVec3("lightColor", 1.0, 1.0, 1.0)

        # Setup light shader
        self.light_shader.use()
        self.light_shader.setVec3("lightColor", 1.0, 1.0, 1.0)

        # Setup plane shader
        self.plane_shader.use()
        self.plane_shader.setVec4("checkerColor1", 1.0, 1.0, 1.0, 1.0)
        self.plane_shader.setVec4("checkerColor2", 0.6, 0.9, 0.6, 1.0)  # pastel green
        model = ke.mat4(1.0)
        model = ke.translate(model, ke.vec3(0.0, 0.0, 0.0))
        #self.plane_shader.setMat4("model", model)
        self.plane_shader.setMat4("model", glm.mat4(1.0))
        self.plane_shader.setVec3("lightColor", 1.0, 1.0, 1.0)

        # Create mesh data
        square_mesh = scene.Prim.create_square_data(1.0)
        sphere_mesh = scene.Prim.create_sphere_data(1.0, 12, 11)
        plane_mesh = scene.Prim.create_plane_data(30.0)

        # Create cube prim
        self.cube_prim = scene.Prim("box", scene.PrimType.Mesh)
        self.cube_prim.set_mesh_data(square_mesh)
        self.cube_prim.add_translate_op(self.box_pos)
        self.addShapePrim(self.cube_shader, self.cube_prim)

        # Create light prim (sphere)
        self.light_prim = scene.Prim("light", scene.PrimType.Mesh)
        self.light_prim.set_mesh_data(sphere_mesh)
        self.light_prim.add_translate_op(self.light_pos)
        self.light_prim.add_scale_op(ke.vec3(self.size, self.size, self.size))
        self.light_prim.add_rotate_quaternion_op(ke.quat(1.0, 0.0, 0.0, 0.0))
        self.addShapePrim(self.light_shader, self.light_prim)

        # Add ground plane (not a prim, just mesh data)
        self.addShape(self.plane_shader, plane_mesh)
        # self.plane_prim = scene.Prim("plane", scene.PrimType.Mesh)
        # self.plane_prim.set_mesh_data(plane_mesh)
        # self.plane_prim.add_translate_op(glm.vec3(0, 0, 0))
        # self.addShapePrim(self.plane_shader, self.plane_prim)

        self.checkError()

    def preRender(self):
        # Update cube color
        self.cube_prim.set_display_color_alpha(
            ke.vec4(self.color[0], self.color[1], self.color[2], self.color[3])
        )
        self.checkError()

    def render(self):
        view = glm.mat4(self.getViewMatrix())
        glm_light_pos = glm.vec3(self.light_pos)

        view_light = glm.vec3(view * glm.vec4(glm_light_pos, 1.0))
        view_cam_pos = glm.vec3(view * glm.vec4(glm.vec3(0, 0, 0), 1.0))
        
        # Update cube shader
        self.cube_shader.use()
        self.cube_shader.setVec3("lightColor", 1.0, 1.0, 1.0)
        self.cube_shader.setVec3("lightPos", view_light)
        self.cube_shader.setVec3("camPos", view_cam_pos)
        
        # Update plane shader
        self.plane_shader.use()
        model = glm.mat4(1.0)
        normalMat = glm.mat3(glm.transpose(glm.inverse(view * model)))
        self.plane_shader.setMat3("normalMat", normalMat)
        self.plane_shader.setVec3("lightPos", view_light)
        self.plane_shader.setVec3("camPos", view_cam_pos)
        self.plane_shader.setVec3("lightColor", 1.0, 1.0, 1.0)

        self.checkError()

    def postRender(self):
        pass

def main():
    app = MyApp()
    app.initialize(SCR_WIDTH, SCR_HEIGHT, False, ke.BackendType.OpenGL)
    app.start()

if __name__ == "__main__":
    main()
