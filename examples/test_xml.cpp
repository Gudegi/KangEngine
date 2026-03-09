#include "animation/robot.hpp"
#include "scene/native/prim.hpp"
#include <Eigen/src/Geometry/Quaternion.h>
#include <glm/fwd.hpp>
#include <imgui.h>
#include <iostream>
#include <memory>

#include "kangEngine.hpp"
#include "material/colors.hpp"
#include "scene/scene_backend.hpp"

using namespace KE;
using namespace KE::Animation;

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

const char* lightFs = R"(
#version 410 core

out vec4 FragColor;

in vec2 TexCoord;
uniform vec3 lightColor;

void main() {
   FragColor = vec4(lightColor, 1.0);
}
)";

const char* stlFs = R"(
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
)";

const char* stlVs = R"(
#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 Normal;
out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMat;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    FragPos = vec3(view * model * vec4(aPos, 1.0f));
    Normal = normalMat * aNormal;
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

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
    FragPos = vec3(view * model * vec4(aPos, 1.0f));
    Normal = normalMat * aNormal;
}
)";

const char* checkerBoardFs = R"(
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
)";

class MyApp : public App {
  public:
    std::unique_ptr<Backend::Shader> stlShader;
    std::unique_ptr<Backend::Shader> lightShader;
    std::unique_ptr<Backend::Shader> planeShader;

    Scene::Prim* lightPrim = nullptr;

    // Robot
    Robot robot;

    const Color& white = ColorLibrary::get(KE::ColorType::WHITE);
    const Color& pastelGreen = ColorLibrary::get(KE::ColorType::PASTEL_GREEN);

    glm::vec3 lightPos = glm::vec3(0.5f, -2.0f, 3.0f); // meters
    float size = 0.05f;                                // light sphere radius
    float deg = 0.f;
    glm::vec3 robotPos = glm::vec3(0.0f, 0.0f, 1.5f);

    void initialize(int width, int height, UpAxis upAxis,
                    Backend::BackendType backendType) {
        KE_TRACE_FUNCTION();
        App::initialize(width, height, false, upAxis, backendType);
    }

    void setup() override {
        KE_TRACE_FUNCTION();
        this->setRenderHz(1.0f / 30.0f);
        stlShader = getGraphicsDevice()->createShader(stlVs, stlFs);
        lightShader = getGraphicsDevice()->createShader(testVs, lightFs);
        planeShader = getGraphicsDevice()->createShader(testVs, checkerBoardFs);

        stlShader->use();
        stlShader->setVec3("lightColor", white.r, white.g, white.b);

        lightShader->use();
        lightShader->setVec3("lightColor", white.r, white.g, white.b);

        planeShader->use();
        planeShader->setVec4("checkerColor1", glm::vec4(1.f, 1.f, 1.f, 1.0f));
        planeShader->setVec4("checkerColor2",
                             glm::vec4(pastelGreen.r, pastelGreen.g,
                                       pastelGreen.b, pastelGreen.a));
        glm::mat4 model = glm::mat4(1.0f);
        planeShader->setMat4("model", model);

        // Light sphere
        auto sphereMeshData = std::make_shared<Scene::MeshData>(
            Scene::Prim::createSphereData(1.0f, 12, 11));
        lightPrim = getScene()->definePrim("/light", Scene::PrimType::Mesh);
        lightPrim->setMeshData(sphereMeshData);
        addShape(lightShader.get(), lightPrim);
        lightPrim->setAttribute("xformOp:translate", lightPos);
        lightPrim->setAttribute("xformOp:scale", glm::vec3(size));

        // Ground plane (Y-up canonical, rotated for current UpAxis)
        auto* groundPrim =
            getScene()->definePrim("/ground", Scene::PrimType::Mesh);
        groundPrim->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(10.f, UpAxis::Z))); // 10m x 10m
        addShape(planeShader.get(), groundPrim);

        // Load MJCF robot
        robot = Robot::fromMJCF(
            KE::getAssetPath("external/retargetted/unitree_h1/unitree_h1.xml"),
            getScene(), stlShader.get(), this);

        checkError();
    }

    void preRender() override { checkError(); }

    void render() override {
        KE_TRACE_FUNCTION();
        ImGui::Begin("Robot Viewer");
        ImGui::Text("Unitree H1 Robot");
        ImGui::Text("Bodies: %d", robot.numBodies());

        ImGui::SliderAngle("asdf", &deg, 0.0f, 180.0f);
        ImGui::End();

        glm::quat q = glm::quat(glm::vec3(glm::radians(deg), 0, 0));
        robot.setRootTranslation(
            Eigen::Vector3f(robotPos.x, robotPos.y, robotPos.z));
        robot.setJointRotation(2, Eigen::Quaternionf(q.w, q.x, q.y, q.z));
        robot.applyPose();

        robotPos.z += 0.01;

        glm::mat4 view = this->getViewMatrix();
        stlShader->use();
        stlShader->setVec3("lightColor", white.r, white.g, white.b);
        stlShader->setVec3("lightPos",
                           glm::vec3(view * glm::vec4(lightPos, 1.0f)));
        glm::vec3 camPos = glm::vec3(0, 0, 0);
        stlShader->setVec3("camPos", glm::vec3(view * glm::vec4(camPos, 1.0f)));

        planeShader->use();
        planeShader->setVec3("lightPos",
                             glm::vec3(view * glm::vec4(lightPos, 1.0f)));
        planeShader->setVec3("camPos",
                             glm::vec3(view * glm::vec4(camPos, 1.0f)));
        planeShader->setVec3("lightColor", white.r, white.g, white.b);

        checkError();
    }

    void postRender() override {}
};

int main() {
    MyApp app;
    Backend::BackendType backend = Backend::BackendType::OpenGL;
    app.initialize(SCR_WIDTH, SCR_HEIGHT, UpAxis::Z, backend);
    app.start();
    return 0;
}
