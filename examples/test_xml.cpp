#include "kangEngine.hpp"
#include "material/colors.hpp"
#include "animation/skel_mesh.hpp"
#include <iostream>
#include <memory>
#include <string>

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
    
    // specular light (view space: camera = origin)
    vec3 V = normalize(-FragPos);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 32.0);
    vec3 specular = 0.5 * spec * lightColor;
    
    FragColor = vColor * vec4(ambient + diffuse + specular, 1.0);
}
)";

class MyApp : public App {
  public:
    std::unique_ptr<Backend::Shader> meshShader;

    float lightColor[3] = {1.0f, 1.0f, 1.0f};
    glm::vec3 lightPos = glm::vec3(-2.0f, 5.0f, 3.0f);
    Animation::SkelMesh robot;

    void initialize(int width, int height, Backend::BackendType backendType) {
        KE_TRACE_FUNCTION();
        App::initialize(width, height, false, UpAxis::Z, backendType);
    }

    void setup() override {
        KE_TRACE_FUNCTION();
        meshShader = getGraphicsDevice()->createShader(phongVs, phongFs);
        meshShader->setUniformBlockBinding("CameraUBO", 0);

        const std::string mjcfPath =
            KE::getAssetPath("external/retargetted/unitree_h1/unitree_h1.xml");
        robot = Animation::SkelMesh::fromMJCF(mjcfPath, getScene());
        for (auto* prim : robot.bodyPrims()) {
            if (prim)
                addShape(meshShader.get(), prim);
        }

        checkError();
    }

    void preRender() override {
        KE_TRACE_FUNCTION();
        auto view = this->getViewMatrix();

        meshShader->use();
        meshShader->setVec3("lightColor", lightColor[0], lightColor[1],
                            lightColor[2]);
        meshShader->setVec3("lightPos",
                            glm::vec3(view * glm::vec4(lightPos, 1.0f)));

        checkError();
    }

    void render() override { KE_TRACE_FUNCTION(); }

    void postRender() override {}
};

int main() {
    MyApp app;
    Backend::BackendType backend = Backend::BackendType::OpenGL;
    app.initialize(SCR_WIDTH, SCR_HEIGHT, backend);
    app.start();
    return 0;
}
