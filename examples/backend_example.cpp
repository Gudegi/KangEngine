///
/// Backend System Example
/// Author Kyungwon Kang, 2024/11
///

#include "kangEngine.hpp"
#include "../src/backend/graphics_factory.hpp"
#include <iostream>
#include <memory>
#include <glm/gtc/matrix_transform.hpp>

using namespace KE;
using namespace KE::Backend;

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

const char* vertexShaderSource = R"(
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
)";

const char* fragmentShaderSource = R"(
#version 410 core

out vec4 FragColor;

in vec2 TexCoord;

uniform vec3 objectColor;
uniform vec3 lightColor;

void main() {
    FragColor = vec4(objectColor * lightColor, 1.0);
}
)";

class BackendApp: public App
{
private:
    float _objectColor[3] = {1.0f, 0.5f, 0.31f};
    float _lightColor[3] = {1.0f, 1.0f, 1.0f};
    float _size = 1.0f;

    // 기존 KE::Shader (주석 처리)
    // std::unique_ptr<KE::Shader> _cubeShader;

    // 새로운 Backend::Shader
    std::unique_ptr<Backend::Shader> _backendShader;

public:
    void initialize(int width, int height, BackendType backendType) {
        std::cout << "Using " << GraphicsFactory::getBackendName(backendType) << " backend" << std::endl;

        // Use App's built-in backend initialization
        App::initialize(width, height, false, backendType);
    }

    void setup() override
    {
        std::cout << "Backend Example Setup" << std::endl;

        // 기존 KE::Shader 방식 (주석 처리)
        // _cubeShader = std::make_unique<KE::Shader>(vertexShaderSource, fragmentShaderSource);

        // 새로운 Backend Shader 방식 - KE::Shader와 동일한 편리한 인터페이스!
        _backendShader = getGraphicsDevice()->createShader(vertexShaderSource, fragmentShaderSource);

        // Create cube using Prim::createSquareData (which creates a quad that can represent a cube face)
        All cubeData = Prim::createSquareData(1.0f);

        // 새로운 Backend::Shader 오버로드 사용 - 타입 캐스팅 문제 해결!
        addShape(_backendShader.get(), std::make_unique<All>(cubeData));

        std::cout << "Backend Example Setup Complete - Using Backend::Shader!" << std::endl;
    }

    void preRender() override
    {
        // App base class now handles backend operations
        // No need to duplicate them here
    }

    void render() override
    {
        // ImGui controls
        ImGui::Begin("Backend Example Controls");
        ImGui::Text("Backend: %s", Backend::GraphicsFactory::getBackendName(getGraphicsDevice()->getBackendType()));
        ImGui::SliderFloat("Size", &_size, 0.1f, 3.0f);
        ImGui::ColorEdit3("Object Color", _objectColor);
        ImGui::ColorEdit3("Light Color", _lightColor);
        ImGui::End();

        // 기존 KE::Shader 방식 (주석 처리)
        // _cubeShader->use();
        // _cubeShader->setVec3("objectColor", glm::vec3(_objectColor[0], _objectColor[1], _objectColor[2]));

        // 새로운 Backend Shader 사용 - KE::Shader와 동일한 API!
        _backendShader->use();  // bind()와 동일
        _backendShader->setVec3("objectColor", glm::vec3(_objectColor[0], _objectColor[1], _objectColor[2]));
        _backendShader->setVec3("lightColor", glm::vec3(_lightColor[0], _lightColor[1], _lightColor[2]));

        // Set model matrix with size scaling
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(_size));
        _backendShader->setMat4("model", model);

        std::cout << "Using Backend Shader - 완벽한 KE::Shader 호환성!" << std::endl;

        // The shape rendering is now handled by App::coreRender()
        // which will automatically render all shapes added via addShape()
    }

    void postRender() override
    {
        // App base class now handles backend operations
        // No need to duplicate them here
    }
};

int main(){
    BackendApp app;

    // You can choose different backends here
    BackendType backend = BackendType::OpenGL;
    // BackendType backend = BackendType::Vulkan;  // Not implemented yet
    // BackendType backend = BackendType::WebGPU;  // Not implemented yet

    app.initialize(SCR_WIDTH, SCR_HEIGHT, backend);
    app.start();
    return 0;
}