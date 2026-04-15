#include "engine/graphics/material/colors.hpp"
#include "physics/physics.hpp"
#include "scene/scene_backend.hpp"
#include <fmt/base.h>
#include <glm/fwd.hpp>
#include <iostream>
#include <memory>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

#include "kangEngine.hpp"

using namespace KE;

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

const char* phongVs = R"(
#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in vec4 aTransform0;
layout(location = 4) in vec4 aTransform1;
layout(location = 5) in vec4 aTransform2;
layout(location = 6) in vec4 aTransform3;
layout(location = 7) in vec4 aInstanceColor;
layout(std140) uniform CameraUBO {
    mat4 view;
    mat4 proj;
};
out vec3 Normal;
out vec3 FragPos;
out vec4 vColor;
void main() {
    mat4 model = mat4(aTransform0, aTransform1, aTransform2, aTransform3);
    mat4 mv = view * model;
    gl_Position = proj * mv * vec4(aPos, 1.0);
    FragPos = vec3(mv * vec4(aPos, 1.0));
    Normal = mat3(mv) * aNormal;
    vColor = aInstanceColor;
}
)";

const char* phongFs = R"(
#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec4 vColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
void main() {
    float ambientStrength = 0.15;
    vec3 ambient = ambientStrength * lightColor;
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightPos - FragPos);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 V = normalize(-FragPos);  // view space: camera = origin
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 32.0);
    vec3 specular = 0.5 * spec * lightColor;
    FragColor = vColor * vec4(ambient + diffuse + specular, 1.0);
}
)";

const char* groundVs = R"(
#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aTransform0;
layout(location = 4) in vec4 aTransform1;
layout(location = 5) in vec4 aTransform2;
layout(location = 6) in vec4 aTransform3;
layout(std140) uniform CameraUBO {
    mat4 view;
    mat4 proj;
};
out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;
void main() {
    mat4 model = mat4(aTransform0, aTransform1, aTransform2, aTransform3);
    mat4 mv = view * model;
    gl_Position = proj * mv * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    FragPos = vec3(mv * vec4(aPos, 1.0));
    Normal = mat3(mv) * aNormal;
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

    float ambientStrength = 0.15;
    vec3 ambient = ambientStrength * lightColor;    

    vec3 N = normalize(Normal);
    vec3 L = normalize(lightPos - FragPos);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor;

    // specular light (view space: camera = origin)
    vec3 V = normalize(-FragPos);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 32.0);
    vec3 specular = 0.5 * spec * lightColor;

    FragColor = objColor * vec4(ambient + diffuse + specular, 1.0);
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

static constexpr int NUM_BOXES = 10;

struct BoxSpawn {
    glm::vec3 pos;
    glm::vec4 color;
};

static const BoxSpawn boxSpawns[NUM_BOXES] = {
    {{0.0f, 8.0f, 0.0f}, {0.8f, 0.3f, 0.02f, 1.0f}},
    {{0.3f, 10.0f, 0.2f}, {0.2f, 0.6f, 0.9f, 1.0f}},
    {{-0.5f, 12.0f, 0.1f}, {0.9f, 0.2f, 0.3f, 1.0f}},
    {{0.1f, 14.0f, -0.3f}, {0.3f, 0.9f, 0.3f, 1.0f}},
    {{-0.2f, 6.0f, -0.1f}, {0.9f, 0.9f, 0.2f, 1.0f}},
    {{0.4f, 16.0f, 0.4f}, {0.6f, 0.2f, 0.8f, 1.0f}},
    {{-0.3f, 9.0f, -0.4f}, {0.2f, 0.8f, 0.7f, 1.0f}},
    {{0.6f, 11.0f, -0.2f}, {1.0f, 0.5f, 0.0f, 1.0f}},
    {{-0.1f, 13.0f, 0.5f}, {0.5f, 0.3f, 0.7f, 1.0f}},
    {{0.2f, 7.0f, -0.5f}, {0.1f, 0.5f, 0.9f, 1.0f}},
};

class MyApp : public App {
  public:
    std::unique_ptr<Backend::Shader> cubeShader;
    std::unique_ptr<Backend::Shader> lightShader;
    std::unique_ptr<Backend::Shader> planeShader;

    float size = 1.3f;
    float lightColor[3] = {1.0f, 1.0f, 1.0f};
    glm::vec3 lightPos = glm::vec3(-2.0f, 3.0f, 1.0f);

    Scene::Prim* boxPrims[NUM_BOXES] = {};
    Scene::Prim* lightPrim = nullptr;
    Scene::Prim* groundPrim = nullptr;
    bool paused = false;
    bool spaceWasPressed = false;

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
        cubeShader = getGraphicsDevice()->createShader(phongVs, phongFs);
        lightShader = getGraphicsDevice()->createShader(phongVs, lightFs);
        planeShader =
            getGraphicsDevice()->createShader(groundVs, checkerBoardFs);

        cubeShader->setUniformBlockBinding("CameraUBO", 0);
        lightShader->setUniformBlockBinding("CameraUBO", 0);
        planeShader->setUniformBlockBinding("CameraUBO", 0);

        // Light shader
        lightShader->use();
        lightShader->setVec3("lightColor", lightColor[0], lightColor[1],
                             lightColor[2]);

        // Plane shader
        const Color& pastelGreen =
            ColorLibrary::get(KE::ColorType::PASTEL_GREEN);
        planeShader->use();
        planeShader->setVec4("checkerColor1", glm::vec4(1.f, 1.f, 1.f, 1.0f));
        planeShader->setVec4("checkerColor2",
                             glm::vec4(pastelGreen.r, pastelGreen.g,
                                       pastelGreen.b, pastelGreen.a));

        // Light sphere
        lightPrim = getScene()->definePrim("/light", Scene::PrimType::Mesh);
        lightPrim->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createSphereData(1.0f, 12, 11)));
        lightPrim->addTranslateOp(lightPos);
        lightPrim->addScaleOp(glm::vec3(size));
        addShape(lightShader.get(), lightPrim);

        // Ground plane
        groundPrim = getScene()->definePrim("/ground", Scene::PrimType::Mesh);
        groundPrim->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(30.f, UpAxis::Y)));
        addShape(planeShader.get(), groundPrim);

        // Create box mesh data (shared by all boxes)
        auto boxMesh = Scene::Prim::createSquareData(1.0f);
        auto boxMeshPtr = std::make_shared<Scene::MeshData>(std::move(boxMesh));

        // Create 10 box prims + physics bodies
        physics.addDefaultGround();
        for (int i = 0; i < NUM_BOXES; ++i) {
            std::string primPath = "/box_" + std::to_string(i);
            auto* prim =
                getScene()->definePrim(primPath, Scene::PrimType::Mesh);
            prim->setMeshData(boxMeshPtr);
            prim->setAttribute("primvars:displaycolorAlpha",
                               boxSpawns[i].color);
            prim->setAttribute("xformOp:translate", boxSpawns[i].pos);

            addShape(cubeShader.get(), prim);
            physics.addBox(boxSpawns[i].pos.x, boxSpawns[i].pos.y,
                           boxSpawns[i].pos.z);
            boxPrims[i] = prim;
        }

        checkError();
    }

    void preRender() override {
        KE_TRACE_FUNCTION();

        auto view = this->getViewMatrix();
        cubeShader->use();
        cubeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                            lightColor[2]);
        cubeShader->setVec3("lightPos",
                            glm::vec3(view * glm::vec4(lightPos, 1.0f)));

        // Spacebar toggle
        bool spaceDown = glfwGetKey(getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasPressed)
            paused = !paused;
        spaceWasPressed = spaceDown;

        if (!paused)
            physics.step();

        // Update box prim transforms from physics
        auto pxScene = physics.getScene();
        PxU32 nbActors = pxScene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
        std::vector<PxActor*> actors(nbActors);
        pxScene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC, actors.data(),
                           nbActors);

        for (PxU32 i = 0; i < nbActors && i < NUM_BOXES; ++i) {
            auto* actor = static_cast<PxRigidActor*>(actors[i]);
            PxTransform pose = actor->getGlobalPose();
            boxPrims[i]->setAttribute("xformOp:translate",
                                      glm::vec3(pose.p.x, pose.p.y, pose.p.z));
            boxPrims[i]->setAttribute(
                "xformOp:rotateQuaternion",
                glm::quat(pose.q.w, pose.q.x, pose.q.y, pose.q.z));
        }

        checkError();
    }

    void render() override {
        KE_TRACE_FUNCTION();
        ImGui::Begin("PhysX Drop Boxes");
        ImGui::Text("%d boxes simulated", NUM_BOXES);
        if (ImGui::SliderFloat("Light Size", &size, 0.5f, 2.0f)) {
            lightPrim->setAttribute("xformOp:scale", glm::vec3(size));
        }
        ImGui::End();

        // Ground lighting
        auto view = this->getViewMatrix();
        planeShader->use();
        planeShader->setVec3("lightPos",
                             glm::vec3(view * glm::vec4(lightPos, 1.0f)));
        planeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                             lightColor[2]);

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
