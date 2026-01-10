
#include <fmt/base.h>

#include <iostream>
#include <memory>

#include "kangEngine.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

using namespace KE;

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

///
/// LearnOpenGL : Lighting - Colors - Phong Shader
///

const char* lightFs = R"(
#version 410 core

out vec4 FragColor;

in vec2 TexCoord;
uniform vec3 lightColor;

void main() {
   FragColor = vec4(lightColor, 1.0);
}
)";

const char* lightVs = R"(
#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec2 TexCoord;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
    Normal = aNormal;
}
)";

const char* testFs = R"(
#version 410 core

out vec4 FragColor;

in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;

uniform sampler2D texture1, texture2;
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
    FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2) * vec4(objColor) * vec4(phong, 1.0f);
}
)";

const char* testVs = R"(
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
uniform bool IS_VIEW_SPACE;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
    if (IS_VIEW_SPACE) {
        FragPos = vec3(view * model * vec4(aPos, 1.0f));
    } else {
        FragPos = vec3(model * vec4(aPos, 1.0f));
    }
    Normal = normalMat * aNormal; 
}
)";

class MyApp : public App {
  public:
    bool IS_VIEW_SPACE = true;
    std::string defaultPath = "./build/assets/";

    std::unique_ptr<Backend::Shader> cubeShader;
    std::unique_ptr<Backend::Shader> lightShader;

    std::unique_ptr<Backend::Texture> texture;
    std::unique_ptr<Backend::Texture> texture2;

    bool drawTriangle = true;
    float size = 0.3f;
    float color[4] = {0.8f, 0.3f, 0.02f, 1.0f};
    float lightColor[3] = {1.0f, 1.0f, 1.0f};
    float lightPo[3] = {-2.0f, 3.0f, 1.0f};
    glm::vec3 lightPos = glm::vec3(lightPo[0], lightPo[1], lightPo[2]);

    void initialize(int width, int height, Backend::BackendType backendType) {
        std::cout << "Using "
                  << Backend::GraphicsFactory::getBackendName(backendType)
                  << " backend" << std::endl;

        // Use App's built-in backend initialization
        App::initialize(width, height, false, backendType);
    }

    void setup() override {
        std::cout << "setUp called" << std::endl;

        cubeShader = getGraphicsDevice()->createShader(testVs, testFs);
        lightShader = getGraphicsDevice()->createShader(testVs, lightFs);
        texture = getGraphicsDevice()->createTexture(
            defaultPath + "textures/crowdEditing.tga", false, GL_REPEAT,
            GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
        texture2 = getGraphicsDevice()->createTexture(
            defaultPath + "textures/awesomeface.png", true, GL_REPEAT,
            GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

        cubeShader->use();
        cubeShader->setColor("objColor", color[0], color[1], color[2],
                             color[3]);
        cubeShader->setInt("texture1", 0);
        cubeShader->setInt("texture2", 1);
        cubeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                            lightColor[2]);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0, -1, 0));
        cubeShader->setMat4("model", model);

        lightShader->use();
        lightShader->setVec3("lightColor", lightColor[0], lightColor[1],
                             lightColor[2]);
        model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(size));
        model = glm::translate(model, lightPos);
        // model = glm::scale(model, glm::vec3(1.0f));
        lightShader->setMat4("model", model);

        // Create mesh data using Scene::Prim
        auto meshData = Scene::Prim::createSquareData(1.0f);
        auto asdf = std::make_shared<Scene::MeshData>(std::move(meshData));
        GLuint s1 = addShape(cubeShader.get(), asdf);
        GLuint s2 = addShape(lightShader.get(), asdf);

        checkError();
        std::cout << "setUp End" << std::endl;
    }

    void preRender() override {
        std::cout << "preRender Called" << std::endl;
        cubeShader->use();
        cubeShader->setColor("color", color[0], color[1], color[2], color[3]);
        texture->bind(0);
        texture2->bind(1);
        checkError();
        std::cout << "preRender End" << std::endl;
    }

    void render() override {
        std::cout << "render Called" << std::endl;
        fmt::print("asdf\n");

        ImGui::Begin("Custom New Panel");
        ImGui::Text("You can create a panel in main loop.");
        ImGui::Checkbox("Draw Triangle", &drawTriangle);
        ImGui::SliderFloat("Size", &size, 0.5f, 2.0f);
        ImGui::ColorEdit4("Color", color);
        ImGui::End();

        lightShader->use();
        lightPos.x = cos(glfwGetTime()) * 3.0f;
        lightPos.y = sin(glfwGetTime()) * 3.0f;
        glm::mat4 model = glm::mat4(1.0f);
        fmt::print("{}\n", glm::to_string(model));
        model = glm::translate(model, lightPos);
        model = glm::scale(model, glm::vec3(size));
        lightShader->setMat4("model", model);

        cubeShader->use();
        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0, -1, 0));
        // model = glm::scale(model, glm::vec3(size));
        cubeShader->setMat4("model", model);
        cubeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                            lightColor[2]);
        cubeShader->setColor("objColor", color[0], color[1], color[2],
                             color[3]);
        cubeShader->setBool("IS_VIEW_SPACE", IS_VIEW_SPACE);
        if (IS_VIEW_SPACE) {
            auto view = this->getViewMatrix();
            glm::mat3 normalMat = glm::mat3(transpose(inverse(view * model)));
            // fmt::print("{}\n", glm::to_string(normalMat));
            // fmt::print("{}\n", glm::to_string(view*model));
            cubeShader->setMat3("normalMat", normalMat);
            cubeShader->setVec3("lightPos",
                                glm::vec3(view * glm::vec4(lightPos, 1.0f)));
            glm::vec3 camPos = glm::vec3(0, 0, 0);
            cubeShader->setVec3("camPos",
                                glm::vec3(view * glm::vec4(camPos, 1.0f)));
        } else {
            glm::mat3 normalMat = glm::mat3(transpose(inverse(model)));
            cubeShader->setMat3("normalMat", normalMat);
            cubeShader->setVec3("lightPos", lightPos);
            Camera& cam = this->getCamera();
            cubeShader->setVec3("camPos", cam.getCameraPos());
        }
        checkError();
        std::cout << "render End" << std::endl;
    }

    void postRender() override {}
};

int main() {
    MyApp app;
    Backend::BackendType backend = Backend::BackendType::OpenGL;
    app.initialize(SCR_WIDTH, SCR_HEIGHT, backend);
    app.start();
    return 0;
}
