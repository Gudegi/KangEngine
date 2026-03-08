#include "camera/camera.hpp"
#include "scene/native/prim.hpp"
#include "utils/load_utils.hpp"
#include <cfloat>
#include <fmt/base.h>
#include <glm/common.hpp>
#include <glm/fwd.hpp>
#include <iostream>
#include <memory>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

#include "kangEngine.hpp"
#include "material/colors.hpp"
#include "physics/physics.hpp"
#include "scene/scene_backend.hpp"

using namespace KE;

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

const char* testFs = R"(
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
    // if (IS_VIEW_SPACE) {
    //     FragPos = vec3(view * model * vec4(aPos, 1.0f));
    // } else {
    //     FragPos = vec3(model * vec4(aPos, 1.0f));
    // }
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

class MyApp : public App {
  public:
    std::unique_ptr<Backend::Shader> cubeShader;
    std::unique_ptr<Backend::Shader> lightShader;
    std::unique_ptr<Backend::Shader> planeShader;
    std::unique_ptr<Backend::Shader> stlShader;

    Scene::Prim* cubePrim = nullptr;
    Scene::Prim* lightPrim = nullptr;
    Scene::Prim* stlPrim = nullptr;

    bool drawTriangle = true;
    float size = 1.3f;
    float color[4] = {0.8f, 0.3f, 0.02f, 1.0f};
    const Color& white = ColorLibrary::get(KE::ColorType::WHITE);
    const Color& pastelGreen = ColorLibrary::get(KE::ColorType::PASTEL_GREEN);
    glm::vec3 lightPos = glm::vec3(-5.0f, 5.0f, 1.0f);
    glm::vec3 boxPos = glm::vec3(0, 10, 0);

    PhysicsWorld physics = PhysicsWorld(PhysicsConfig());

    void initialize(int width, int height, Backend::BackendType backendType) {
        KE_TRACE_FUNCTION();
        std::cout << "Using "
                  << Backend::GraphicsFactory::getBackendName(backendType)
                  << " backend" << std::endl;

        App::initialize(width, height, false, UpAxis::Y, backendType);
    }

    void setup() override {
        KE_TRACE_FUNCTION();
        cubeShader = getGraphicsDevice()->createShader(testVs, testFs);
        stlShader = getGraphicsDevice()->createShader(stlVs, stlFs);
        lightShader = getGraphicsDevice()->createShader(testVs, lightFs);
        planeShader = getGraphicsDevice()->createShader(testVs, checkerBoardFs);

        cubeShader->use();
        cubeShader->setColor("objColor", color[0], color[1], color[2],
                             color[3]);
        cubeShader->setVec3("lightColor", white.r, white.g, white.b);

        stlShader->use();
        stlShader->setColor("objColor", color[0], color[1], color[2], color[3]);
        stlShader->setVec3("lightColor", white.r, white.g, white.b);

        lightShader->use();
        lightShader->setVec3("lightColor", white.r, white.g, white.b);

        planeShader->use();
        planeShader->setVec4("checkerColor1", glm::vec4(1.f, 1.f, 1.f, 1.0f));
        planeShader->setVec4("checkerColor2",
                             glm::vec4(pastelGreen.r, pastelGreen.g,
                                       pastelGreen.b, pastelGreen.a));
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0, 0, 0));
        planeShader->setMat4("model", model);

        // MeshDatas
        auto squareMeshData = std::make_shared<Scene::MeshData>(
            Scene::Prim::createSquareData(1.0f));
        auto sphereMeshData = std::make_shared<Scene::MeshData>(
            Scene::Prim::createSphereData(1.0f, 12, 11));
        std::string inputfile = KE::getAssetPath(
            "external/retargetted/unitree_h1/assets/pelvis.stl");
        auto stlMeshData =
            std::make_shared<Scene::MeshData>(loadStl(inputfile));
        stlMeshData->fillMissingAttributes();
        fmt::print("STL loaded: vertices={}, normals={}, uvs={}, indices={}\n",
                   stlMeshData->vertices.size(), stlMeshData->normals.size(),
                   stlMeshData->uvs.size(), stlMeshData->indices.size());

        // Debug: print bounding box of STL mesh
        glm::vec3 minBound(FLT_MAX), maxBound(-FLT_MAX);
        for (const auto& v : stlMeshData->vertices) {
            minBound = glm::min(minBound, v);
            maxBound = glm::max(maxBound, v);
        }
        fmt::print("STL bounds: min=({},{},{}), max=({},{},{})\n", minBound.x,
                   minBound.y, minBound.z, maxBound.x, maxBound.y, maxBound.z);

        // prim1
        cubePrim = getScene()->definePrim("box", Scene::PrimType::Mesh);
        cubePrim->setMeshData(squareMeshData);
        GLuint s1 = addShape(cubeShader.get(), cubePrim);
        cubePrim->setAttribute("xformOp:translate", boxPos);

        // prim2
        lightPrim = getScene()->definePrim("light", Scene::PrimType::Mesh);
        lightPrim->setMeshData(sphereMeshData);
        GLuint s2 = addShape(lightShader.get(), lightPrim);
        lightPrim->setAttribute("xformOp:translate", lightPos);
        lightPrim->setAttribute("xformOp:scale", glm::vec3(size));
        lightPrim->setAttribute("xformOp:rotateQuaternion",
                                glm::quat(1, 0, 0, 0));
        // fmt::print("lightPrim address: {}\n", (void*)lightPrim.get());

        // prim3
        stlPrim = getScene()->definePrim("stlMesh", Scene::PrimType::Mesh);
        stlPrim->setMeshData(stlMeshData);
        stlPrim->setAttribute("xformOp:translate", boxPos);
        stlPrim->setAttribute("xformOp:scale", glm::vec3(30.0f));
        GLuint s4 = addShape(stlShader.get(), stlPrim);

        // not prim
        auto planeMeshData = std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(30.f));
        GLuint s3 = addShape(planeShader.get(), planeMeshData);

        physics.addDefaultGround();
        physics.addBox(boxPos.x, boxPos.y, boxPos.z);

        checkError();
    }

    void preRender() override {
        KE_TRACE_FUNCTION();
        cubePrim->setAttribute(
            "primvars:displaycolorAlpha",
            glm::vec4(color[0], color[1], color[2], color[3]));
        stlPrim->setAttribute(
            "primvars:displaycolorAlpha",
            glm::vec4(color[0], color[1], color[2], color[3]));
        physics.step();
        checkError();
    }

    void render() override {
        KE_TRACE_FUNCTION();
        ImGui::Begin("Custom New Panel");
        ImGui::Text("You can create a panel in main loop.");
        ImGui::Checkbox("Draw Triangle", &drawTriangle);
        ImGui::SliderFloat("Size", &size, 0.5f, 2.0f);
        ImGui::ColorEdit4("Color", color);
        ImGui::End();

        lightPrim->setAttribute("xformOp:scale", glm::vec3(size, size, size));

        auto scene = physics.getScene();
        PxU32 nbActors = scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
        if (nbActors) {
            // fmt::print("Num of Dynamic actors : {}\n", nbActors);
        }
        PxActor* actors[1];
        scene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC, actors, 1);
        PxRigidActor* actor = static_cast<PxRigidActor*>(actors[0]);
        PxTransform pose = actor->getGlobalPose();

        boxPos = glm::vec3(pose.p.x, pose.p.y, pose.p.z);
        glm::quat boxQuat = glm::quat(pose.q.w, pose.q.x, pose.q.y, pose.q.z);

        cubePrim->setAttribute("xformOp:translate", boxPos);
        cubePrim->setAttribute("xformOp:rotateQuaternion", boxQuat);

        stlPrim->setAttribute("xformOp:translate", boxPos);
        stlPrim->setAttribute("xformOp:rotateQuaternion", boxQuat);

        glm::mat4 view = this->getViewMatrix();
        cubeShader->use();
        cubeShader->setVec3("lightColor", white.r, white.g, white.b);
        cubeShader->setVec3("lightPos",
                            glm::vec3(view * glm::vec4(lightPos, 1.0f)));
        glm::vec3 camPos = glm::vec3(0, 0, 0);
        cubeShader->setVec3("camPos",
                            glm::vec3(view * glm::vec4(camPos, 1.0f)));

        stlShader->use();
        stlShader->setVec3("lightColor", white.r, white.g, white.b);
        stlShader->setVec3("lightPos",
                           glm::vec3(view * glm::vec4(lightPos, 1.0f)));
        stlShader->setVec3("camPos", glm::vec3(view * glm::vec4(camPos, 1.0f)));

        // Ground
        planeShader->use();
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat3 normalMat = glm::mat3(transpose(inverse(view * model)));
        planeShader->setMat3("normalMat", normalMat);
        planeShader->setVec3("lightPos",
                             glm::vec3(view * glm::vec4(lightPos, 1.0f)));
        planeShader->setVec3("lightColor", white.r, white.g, white.b);

        checkError();
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
