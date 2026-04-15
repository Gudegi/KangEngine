#include "engine/scene/native/prim.hpp"
#include <fmt/base.h>
#include <glm/fwd.hpp>
#include <iostream>
#include <memory>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

#include "kangEngine.hpp"
#include "engine/graphics/material/colors.hpp"
#include "physics/physics.hpp"
#include "engine/scene/scene_backend.hpp"

using namespace KE;

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

const char* testVs = R"(
#version 410 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

// Per-instance transform (mat4 → vec4 × 4)
layout(location = 3) in vec4 aTransform0;
layout(location = 4) in vec4 aTransform1;
layout(location = 5) in vec4 aTransform2;
layout(location = 6) in vec4 aTransform3;
layout(location = 7) in vec4 aInstanceColor;

layout(std140) uniform CameraUBO {
    mat4 view;
    mat4 proj;
};

out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;
out vec4 vColor;

void main() {
    mat4 model = mat4(aTransform0, aTransform1, aTransform2, aTransform3);
    mat4 mv = view * model;
    gl_Position = proj * mv * vec4(aPos, 1.0);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
    FragPos = vec3(mv * vec4(aPos, 1.0));
    Normal = mat3(mv) * aNormal;  // uniform scale 가정
    vColor = aInstanceColor;
}
)";

const char* testFs = R"(
#version 410 core

out vec4 FragColor;

in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;
in vec4 vColor;

uniform vec3 lightColor;
uniform vec3 lightPos;

void main() {
    // ambient
    float ambientStrength = 0.15;
    vec3 ambient = ambientStrength * lightColor;

    // diffuse
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightPos - FragPos);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor;

    // specular (view space: camera = origin)
    vec3 V = normalize(-FragPos);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 32.0);
    vec3 specular = 0.5 * spec * lightColor;

    FragColor = vColor * vec4(ambient + diffuse + specular, 1.0);
}
)";

const char* lightFs = R"(
#version 410 core

out vec4 FragColor;

uniform vec3 lightColor;

void main() {
   FragColor = vec4(lightColor, 1.0);
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

float checkerboardPattern(vec2 uv) {
    return mod(floor(uv.x * 10.0) + floor(uv.y * 10.0), 2.0);
}

void main() {
    float t = checkerboardPattern(TexCoord);
    vec4 objColor = mix(checkerColor1, checkerColor2, t);

    // ambient
    float ambientStrength = 0.15;
    vec3 ambient = ambientStrength * lightColor;

    // diffuse
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightPos - FragPos);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor;

    // specular (view space: camera = origin)
    vec3 V = normalize(-FragPos);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 32.0);
    vec3 specular = 0.5 * spec * lightColor;

    FragColor = objColor * vec4(ambient + diffuse + specular, 1.0);
}
)";

class MyApp : public App {
  public:
    std::unique_ptr<Backend::Shader> cubeShader;
    std::unique_ptr<Backend::Shader> lightShader;
    std::unique_ptr<Backend::Shader> planeShader;

    // TODO: stage가 갖고있게?
    std::shared_ptr<Scene::Prim> cubePrim;
    std::shared_ptr<Scene::Prim> lightPrim;
    std::shared_ptr<Scene::Prim> bunnyPrim;

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

        App::initialize(width, height, false, backendType);
    }

    void setup() override {
        KE_TRACE_FUNCTION();
        cubeShader = getGraphicsDevice()->createShader(testVs, testFs);
        lightShader = getGraphicsDevice()->createShader(testVs, lightFs);
        planeShader = getGraphicsDevice()->createShader(testVs, checkerBoardFs);

        cubeShader->setUniformBlockBinding("CameraUBO", 0);
        lightShader->setUniformBlockBinding("CameraUBO", 0);
        planeShader->setUniformBlockBinding("CameraUBO", 0);

        lightShader->use();
        lightShader->setVec3("lightColor", white.r, white.g, white.b);

        planeShader->use();
        planeShader->setVec4("checkerColor1", glm::vec4(1.f, 1.f, 1.f, 1.0f));
        planeShader->setVec4("checkerColor2",
                             glm::vec4(pastelGreen.r, pastelGreen.g,
                                       pastelGreen.b, pastelGreen.a));

        // MeshDatas
        auto squareMeshData = std::make_shared<Scene::MeshData>(
            Scene::Prim::createSquareData(1.0f));
        auto sphereMeshData = std::make_shared<Scene::MeshData>(
            Scene::Prim::createSphereData(1.0f, 12, 11));
        std::string inputfile = KE::getAssetPath("external/bunny.obj");
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; // Path to material files
        auto bunnyMeshData = std::make_shared<Scene::MeshData>(
            loadObj(inputfile, reader_config));

        // prim1
        cubePrim = std::make_shared<Scene::Prim>("box", Scene::PrimType::Mesh);
        cubePrim->setMeshData(squareMeshData);
        GLuint s1 = addShape(cubeShader.get(), cubePrim.get());
        cubePrim->setAttribute("xformOp:translate", boxPos);

        // prim2
        lightPrim =
            std::make_shared<Scene::Prim>("light", Scene::PrimType::Mesh);
        lightPrim->setMeshData(sphereMeshData);
        GLuint s2 = addShape(lightShader.get(), lightPrim.get());
        lightPrim->setAttribute("xformOp:translate", lightPos);
        lightPrim->setAttribute("xformOp:scale", glm::vec3(size));
        lightPrim->setAttribute("xformOp:rotateQuaternion",
                                glm::quat(1, 0, 0, 0));
        // fmt::print("lightPrim address: {}\n", (void*)lightPrim.get());

        // prim3
        bunnyPrim =
            std::make_shared<Scene::Prim>("bunny", Scene::PrimType::Mesh);
        bunnyPrim->setMeshData(bunnyMeshData);
        bunnyPrim->setAttribute("xformOp:translate", boxPos);
        bunnyPrim->setAttribute("xformOp:scale", glm::vec3(size));
        GLuint s4 = addShape(cubeShader.get(), bunnyPrim.get());

        auto* planePrim =
            getScene()->definePrim("plane", Scene::PrimType::Mesh);
        planePrim->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(30.f)));
        addShape(planeShader.get(), planePrim);

        physics.addDefaultGround();
        physics.addBox(boxPos.x, boxPos.y, boxPos.z);

        checkError();
    }

    void preRender() override {
        KE_TRACE_FUNCTION();
        cubePrim->setAttribute(
            "primvars:displaycolorAlpha",
            glm::vec4(color[0], color[1], color[2], color[3]));
        bunnyPrim->setAttribute(
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

        bunnyPrim->setAttribute("xformOp:translate", boxPos);
        bunnyPrim->setAttribute("xformOp:rotateQuaternion", boxQuat);

        // lightPos를 view space로 변환해서 lighting uniform 세팅
        glm::mat4 view = this->getViewMatrix();
        glm::vec3 lightPosView = glm::vec3(view * glm::vec4(lightPos, 1.0f));

        cubeShader->use();
        cubeShader->setVec3("lightPos", lightPosView);
        cubeShader->setVec3("lightColor", white.r, white.g, white.b);

        planeShader->use();
        planeShader->setVec3("lightPos", lightPosView);
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
