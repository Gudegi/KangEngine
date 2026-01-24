import sys
import os
sys.path.append(os.getcwd() + "/build")
import kangengine

# Shader source code (same as backend_light0.cpp)
lightFs = """
#version 410 core

out vec4 FragColor;

in vec2 TexCoord;
uniform vec3 lightColor;

void main() {
   FragColor = vec4(lightColor, 1.0);
}
"""

lightVs = """
#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
}
"""

testFs = """
#version 410 core

out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1, texture2;
uniform vec4 objColor;
uniform vec3 lightColor;

void main() {
   FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2) * vec4(objColor) * vec4(lightColor, 1.0f);
}
"""

testVs = """
#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
}
"""

class MyPyApp(kangengine.App):
    def __init__(self):
        super().__init__()
        self.defaultPath = "build/assets/"
        self.cubeShader = None
        self.lightShader = None
        self.texture = None
        self.texture2 = None
        self.size = 0.7
        self.color = [0.8, 0.3, 0.02, 1.0]
        self.lightColor = [1.0, 1.0, 1.0]

    def setup(self):
        print("Python setup called!")

        # Create mesh data
        meshData = kangengine.createSquareData(1.0)

        # Create shaders using backend device
        device = self.getGraphicsDevice()
        self.cubeShader = device.createShader(testVs, testFs)
        self.lightShader = device.createShader(lightVs, lightFs)

        # Create textures
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        basePath = os.path.join(project_root, "build/assets/")

        texturePath1 = os.path.join(basePath, "textures/crowdEditing.tga")
        texturePath2 = os.path.join(basePath, "textures/awesomeface.png")

        self.texture = device.createTexture(
            texturePath1,
            False, 0x2901, 0x2703, 0x2601  # GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR
        )
        self.texture2 = device.createTexture(
            texturePath2,
            True, 0x2901, 0x2703, 0x2601
        )

        # Setup cube shader
        self.cubeShader.use()
        self.cubeShader.setFloat("size", self.size)
        self.cubeShader.setColor("objColor", self.color[0], self.color[1], self.color[2], self.color[3])
        self.cubeShader.setInt("texture1", 0)
        self.cubeShader.setInt("texture2", 1)
        self.cubeShader.setVec3("lightColor", self.lightColor[0], self.lightColor[1], self.lightColor[2])

        # Set model matrix for cube (translate down and scale)
        model = kangengine.mat4(1.0)  # Identity matrix
        model = kangengine.translate(model, kangengine.vec3(0, -1, 0))
        model = kangengine.scale(model, kangengine.vec3(self.size, self.size, self.size))
        self.cubeShader.setMat4("model", model)

        # Setup light shader
        self.lightShader.use()
        self.lightShader.setVec3("lightColor", self.lightColor[0], self.lightColor[1], self.lightColor[2])

        # Set model matrix for light (identity)
        model = kangengine.mat4(1.0)
        self.lightShader.setMat4("model", model)

        # Add shapes to renderer
        self.addShape(self.cubeShader, meshData)
        self.addShape(self.lightShader, meshData)

        self.checkError()
        print("Python setup complete!")

    def preRender(self):
        # Bind textures
        self.cubeShader.use()
        self.cubeShader.setColor("objColor", self.color[0], self.color[1], self.color[2], self.color[3])
        if self.texture:
            self.texture.bind(0)
        if self.texture2:
            self.texture2.bind(1)
        self.checkError()

    def render(self):
        # Update shader with current values
        self.cubeShader.use()

        # Update model matrix with current size
        model = kangengine.mat4(1.0)
        model = kangengine.translate(model, kangengine.vec3(0, -1, 0))
        model = kangengine.scale(model, kangengine.vec3(self.size, self.size, self.size))
        self.cubeShader.setMat4("model", model)

        self.cubeShader.setColor("objColor", self.color[0], self.color[1], self.color[2], self.color[3])
        self.checkError()

    def postRender(self):
        pass


if __name__ == "__main__":
    print("="*60)
    print("KangEngine Python Backend Example")
    print("="*60)

    # Create Python App instance that replicates backend_light0.cpp
    app = MyPyApp()
    app.initialize(1920, 1080, False, kangengine.BackendType.OpenGL)
    app.start()

