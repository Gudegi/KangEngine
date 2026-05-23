#include "kangEngine.hpp"
#include "engine/graphics/material/colors.hpp"
#include <iostream>
#include <memory>
#include "asset/mesh_loader.hpp"
#include <tiny_obj_loader.h>

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
    return mod(floor(uv.x) + floor(uv.y), 2.0);
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

class MyApp : public App {
  public:
    std::unique_ptr<Backend::Shader> meshShader;
    std::unique_ptr<Backend::Shader> planeShader;

    float lightColor[3] = {1.0f, 1.0f, 1.0f};
    glm::vec3 lightPos = glm::vec3(-2.0f, 5.0f, 3.0f);

    Scene::Prim* stlPrim = nullptr;
    Scene::Prim* objPrim = nullptr;
    Scene::Prim* groundPrim = nullptr;

    void initialize(int width, int height, Backend::BackendType backendType) {
        KE_TRACE_FUNCTION();
        App::initialize(width, height, false, UpAxis::Y, backendType);
    }

    void setup() override {
        KE_TRACE_FUNCTION();
        meshShader = getGraphicsDevice()->createShader(phongVs, phongFs);
        planeShader =
            getGraphicsDevice()->createShader(groundVs, checkerBoardFs);

        meshShader->setUniformBlockBinding("CameraUBO", 0);
        planeShader->setUniformBlockBinding("CameraUBO", 0);

        const Color& pastelGreen =
            ColorLibrary::get(KE::ColorType::PASTEL_GREEN);
        planeShader->use();
        planeShader->setVec4("checkerColor1", glm::vec4(1.f, 1.f, 1.f, 1.0f));
        planeShader->setVec4("checkerColor2",
                             glm::vec4(pastelGreen.r, pastelGreen.g,
                                       pastelGreen.b, pastelGreen.a));

        groundPrim = getScene()->definePrim("/ground", Scene::PrimType::Mesh);
        groundPrim->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(30.f, UpAxis::Y)));
        addShape(planeShader.get(), groundPrim);

        std::string stlPath =
            std::string(KANGENGINE_ASSETS_ROOT) +
            "/external/retargetted/unitree_h1/assets/pelvis.stl";

        try {
            auto stlData = Asset::loadStl(stlPath);
            auto stlMeshPtr =
                std::make_shared<Scene::MeshData>(std::move(stlData));

            stlPrim =
                getScene()->definePrim("/stl_mesh", Scene::PrimType::Mesh);
            stlPrim->setMeshData(stlMeshPtr);

            // 색상과 초기 변환 설정
            stlPrim->setAttribute("primvars:displaycolorAlpha",
                                  glm::vec4(0.8f, 0.4f, 0.1f, 1.0f));
            stlPrim->addScaleOp(glm::vec3(10.f));
            stlPrim->addTranslateOp(glm::vec3(0.0f, 1.0f, 0.0f));

            addShape(meshShader.get(), stlPrim);
        } catch (const std::exception& e) {
            std::cerr << "STL 로드 실패: " << e.what() << "\n";
        }

        std::string objPath =
            std::string(KANGENGINE_ASSETS_ROOT) + "/external/bunny.obj";

        try {
            tinyobj::ObjReaderConfig reader_config;
            reader_config.mtl_search_path =
                std::string(KANGENGINE_ASSETS_ROOT) + "/external/";
            auto objData = Asset::loadObj(objPath, reader_config);
            auto objMeshPtr =
                std::make_shared<Scene::MeshData>(std::move(objData));

            objPrim =
                getScene()->definePrim("/obj_mesh", Scene::PrimType::Mesh);
            objPrim->setMeshData(objMeshPtr);

            objPrim->setAttribute("primvars:displaycolorAlpha",
                                  glm::vec4(0.2f, 0.6f, 0.9f, 1.0f));
            objPrim->addScaleOp(glm::vec3(1.f));
            objPrim->addTranslateOp(
                glm::vec3(2.0f, 1.0f, 0.0f)); // STL 옆에 배치

            addShape(meshShader.get(), objPrim);
        } catch (const std::exception& e) {
            std::cerr << "OBJ 로드 실패: " << e.what() << "\n";
        }

        checkError();
    }

    void preRender() override {
        KE_TRACE_FUNCTION();
        auto view = this->getViewMatrix();

        // 렌더링 전 조명 위치 (View Space 변환) 업데이트
        meshShader->use();
        meshShader->setVec3("lightColor", lightColor[0], lightColor[1],
                            lightColor[2]);
        meshShader->setVec3("lightPos",
                            glm::vec3(view * glm::vec4(lightPos, 1.0f)));

        planeShader->use();
        planeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                             lightColor[2]);
        planeShader->setVec3("lightPos",
                             glm::vec3(view * glm::vec4(lightPos, 1.0f)));

        checkError();
    }

    void render() override {
        KE_TRACE_FUNCTION();
        // ImGui 창으로 간단한 디버그 UI 생성
        ImGui::Begin("Mesh Viewer");
        ImGui::Text("STL & OBJ Rendering Test");
        if (stlPrim) {
            static float scale = 1.0f;
            if (ImGui::SliderFloat("STL Scale", &scale, 0.1f, 10.0f)) {
                stlPrim->setAttribute("xformOp:scale", glm::vec3(scale));
            }
        }
        if (objPrim) {
            static float objScale = 10.0f;
            if (ImGui::SliderFloat("OBJ Scale", &objScale, 0.1f, 20.0f)) {
                objPrim->setAttribute("xformOp:scale", glm::vec3(objScale));
            }
        }
        ImGui::End();
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
