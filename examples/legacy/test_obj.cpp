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

class MyApp : public App {
  public:
    std::unique_ptr<Backend::Shader> cubeShader;
    std::unique_ptr<Backend::Shader> lightShader;
    std::unique_ptr<Backend::Shader> planeShader;

    bool drawTriangle = true;
    float size = 1.3f;
    float color[4] = {0.8f, 0.3f, 0.02f, 1.0f};
    float lightColor[3] = {1.0f, 1.0f, 1.0f};
    float lightPo[3] = {-2.0f, 3.0f, 1.0f};
    glm::vec3 lightPos = glm::vec3(lightPo[0], lightPo[1], lightPo[2]);
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

        cubeShader->use();
        cubeShader->setColor("objColor", color[0], color[1], color[2],
                             color[3]);
        cubeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                            lightColor[2]);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, boxPos);
        cubeShader->setMat4("model", model);

        lightShader->use();
        lightShader->setVec3("lightColor", lightColor[0], lightColor[1],
                             lightColor[2]);
        model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(size));
        model = glm::translate(model, lightPos);
        lightShader->setMat4("model", model);

        const Color& pastelPink =
            ColorLibrary::get(KE::ColorType::PASTEL_GREEN);
        planeShader->use();
        planeShader->setVec4("checkerColor1", glm::vec4(1.f, 1.f, 1.f, 1.0f));
        planeShader->setVec4(
            "checkerColor2",
            glm::vec4(pastelPink.r, pastelPink.g, pastelPink.b, pastelPink.a));

        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0, 0, 0));
        planeShader->setMat4("model", model);

        // Create mesh data using Scene::Prim
        auto meshData = Scene::Prim::createSquareData(1.0f);
        auto asdf = std::make_shared<Scene::MeshData>(std::move(meshData));

        auto asdf3 = std::make_shared<Scene::MeshData>(
            std::move(Scene::Prim::createSphereData(1.0f, 12, 11)));
        GLuint s1 = addShape(cubeShader.get(), asdf);
        GLuint s2 = addShape(lightShader.get(), asdf3);

        auto planeData = Scene::Prim::createPlaneData(30.f);
        auto planeMesh =
            std::make_shared<Scene::MeshData>(std::move(planeData));
        GLuint s3 = addShape(planeShader.get(), planeMesh);

        physics.addDefaultGround();
        physics.addBox(boxPos.x, boxPos.y, boxPos.z);

        std::string inputfile = "./build/assets/external/bunny.obj";
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; // Path to material files
        auto meshData2 = loadObj(inputfile, reader_config);
        auto asdf2 = std::make_shared<Scene::MeshData>(std::move(meshData2));
        GLuint s4 = addShape(cubeShader.get(), asdf2);

        checkError();
    }

    void preRender() override {
        KE_TRACE_FUNCTION();
        cubeShader->use();
        cubeShader->setColor("objColor", color[0], color[1], color[2],
                             color[3]);
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

        lightShader->use();
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(size));
        model = glm::translate(model, lightPos);
        lightShader->setMat4("model", model);

        cubeShader->use();
        model = glm::mat4(1.0f);

        auto scene = physics.getScene();
        PxU32 nbActors = scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
        if (nbActors) {
            fmt::print("Num of Dynamic actors : {}\n", nbActors);
        }

        PxActor* actors[1];
        scene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC, actors, 1);
        PxRigidActor* actor = static_cast<PxRigidActor*>(actors[0]);
        PxTransform pose = actor->getGlobalPose();

        model = glm::translate(model, glm::vec3(pose.p.x, pose.p.y, pose.p.z));
        fmt::print("model : {}\n",
                   glm::to_string(glm::vec3(pose.p.x, pose.p.y, pose.p.z)));
        // model = glm::scale(model, glm::vec3(size));
        cubeShader->setMat4("model", model);
        cubeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                            lightColor[2]);
        cubeShader->setColor("objColor", color[0], color[1], color[2],
                             color[3]);

        auto view = this->getViewMatrix();
        glm::mat3 normalMat = glm::mat3(transpose(inverse(view * model)));
        cubeShader->setMat3("normalMat", normalMat);
        cubeShader->setVec3("lightPos",
                            glm::vec3(view * glm::vec4(lightPos, 1.0f)));
        glm::vec3 camPos = glm::vec3(0, 0, 0);
        cubeShader->setVec3("camPos",
                            glm::vec3(view * glm::vec4(camPos, 1.0f)));

        // Ground
        planeShader->use();
        model = glm::mat4(1.0f);
        normalMat = glm::mat3(transpose(inverse(view * model)));
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
