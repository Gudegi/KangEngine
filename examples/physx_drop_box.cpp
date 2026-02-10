#include "material/colors.hpp"
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
        cubeShader = getGraphicsDevice()->createShader(testVs, testFs);
        lightShader = getGraphicsDevice()->createShader(testVs, lightFs);
        planeShader = getGraphicsDevice()->createShader(testVs, checkerBoardFs);

        // Light shader
        lightShader->use();
        lightShader->setVec3("lightColor", lightColor[0], lightColor[1],
                             lightColor[2]);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(size));
        model = glm::translate(model, lightPos);
        lightShader->setMat4("model", model);

        // Plane shader
        const Color& pastelGreen =
            ColorLibrary::get(KE::ColorType::PASTEL_GREEN);
        planeShader->use();
        planeShader->setVec4("checkerColor1", glm::vec4(1.f, 1.f, 1.f, 1.0f));
        planeShader->setVec4("checkerColor2",
                             glm::vec4(pastelGreen.r, pastelGreen.g,
                                       pastelGreen.b, pastelGreen.a));
        planeShader->setMat4("model", glm::mat4(1.0f));

        // Light sphere mesh (raw, no prim)
        auto cubeMesh = Scene::Prim::createSquareData(1.0f);
        auto cubeMeshPtr =
            std::make_shared<Scene::MeshData>(std::move(cubeMesh));
        addShape(lightShader.get(), cubeMeshPtr);

        // Ground plane mesh (raw, no prim)
        auto planeData = Scene::Prim::createPlaneData(30.f);
        auto planeMesh =
            std::make_shared<Scene::MeshData>(std::move(planeData));
        addShape(planeShader.get(), planeMesh);

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

        // Set cube shader lighting (before coreRender draws prims)
        auto view = this->getViewMatrix();
        glm::vec3 camPos = glm::vec3(0, 0, 0);

        cubeShader->use();
        cubeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                            lightColor[2]);
        cubeShader->setVec3("lightPos",
                            glm::vec3(view * glm::vec4(lightPos, 1.0f)));
        cubeShader->setVec3("camPos",
                            glm::vec3(view * glm::vec4(camPos, 1.0f)));

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
        ImGui::SliderFloat("Light Size", &size, 0.5f, 2.0f);
        ImGui::End();

        lightShader->use();
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, lightPos);
        model = glm::scale(model, glm::vec3(size));
        lightShader->setMat4("model", model);

        // Ground lighting
        auto view = this->getViewMatrix();
        planeShader->use();
        glm::mat4 groundModel = glm::mat4(1.0f);
        glm::mat3 normalMat = glm::mat3(transpose(inverse(view * groundModel)));
        planeShader->setMat3("normalMat", normalMat);
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
