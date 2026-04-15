///
/// H1 Robot — PhysX Articulation + PD Joint Control
///
/// Loads Unitree H1 from MJCF for rendering (via Robot class) and builds a
/// matching PxArticulationReducedCoordinate for physics simulation.
/// Each joint is a revolute drive with configurable PD gains.
///
/// Controls:
///   Space : pause / resume
///   WASD + mouse : camera
///   ImGui : PD gains, per-joint target angles
///

#include "animation/skel_mesh.hpp"
#include "animation/skeleton_math.hpp"
#include "kangEngine.hpp"
#include "material/colors.hpp"
#include "physics/articulation.hpp"
#include "physics/physics.hpp"
#include <Eigen/Geometry>
#include <glm/glm.hpp>
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>

using namespace KE;
using namespace KE::Animation;
using namespace physx;

// ---------------------------------------------------------------------------
// Shaders (copied from test_xml.cpp)
// ---------------------------------------------------------------------------

static const char* stlVs = R"(
#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in vec4 aInstanceTransform0;
layout(location = 4) in vec4 aInstanceTransform1;
layout(location = 5) in vec4 aInstanceTransform2;
layout(location = 6) in vec4 aInstanceTransform3;
layout(location = 7) in vec4 aInstanceColor;

layout(std140) uniform cameraUBO {
    mat4 view;
    mat4 projection;
};

out vec3 Normal;
out vec3 FragPos;
out vec4 vColor;

void main() {
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(view * model * vec4(aPos, 1.0));
    Normal = mat3(view * model) * aNormal;
    vColor = aInstanceColor;
}
)";

static const char* stlFs = R"(
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
    float specularStrength = 0.5;
    vec3 V = normalize(-FragPos);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    FragColor = vec4(vColor.rgb * (ambient + diffuse + specular), vColor.a);
}
)";

static const char* groundVs = R"(
#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aInstanceTransform0;
layout(location = 4) in vec4 aInstanceTransform1;
layout(location = 5) in vec4 aInstanceTransform2;
layout(location = 6) in vec4 aInstanceTransform3;

layout(std140) uniform cameraUBO {
    mat4 view;
    mat4 projection;
};

out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;

void main() {
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    FragPos = vec3(view * model * vec4(aPos, 1.0));
    Normal = mat3(view * model) * aNormal;
}
)";

static const char* groundFs = R"(
#version 410 core
out vec4 FragColor;
in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;
uniform vec3 lightColor;
uniform vec3 lightPos;
float checker(vec2 uv) {
    return mod(floor(uv.x * 8.0) + floor(uv.y * 8.0), 2.0);
}
void main() {
    float t = checker(TexCoord);
    vec4 col = mix(vec4(0.85, 0.85, 0.85, 1.0), vec4(0.55, 0.75, 0.55, 1.0), t);
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightPos - FragPos);
    float diff = max(dot(N, L), 0.0);
    vec3 light = 0.2 * lightColor + diff * lightColor;
    FragColor = col * vec4(light, 1.0);
}
)";

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

class H1PhysicsApp : public App {
  public:
    std::unique_ptr<Backend::Shader> stlShader;
    std::unique_ptr<Backend::Shader> groundShader;

    glm::vec3 lightPos = {0.f, -2.f, 4.f};

    SkelMesh robot;
    PhysicsWorld physics{PhysicsConfig::zUp()};
    Articulation artic;

    std::vector<float> targets; // joint target angles, size = n-1

    float kp = 200.f;
    float kd = 20.f;
    bool paused = false;
    bool spaceWasDown = false;

    // -----------------------------------------------------------------------
    void setup() override {
        stlShader = getGraphicsDevice()->createShader(stlVs, stlFs);
        groundShader = getGraphicsDevice()->createShader(groundVs, groundFs);

        stlShader->use();
        stlShader->setUniformBlockBinding("cameraUBO", 0);
        stlShader->setVec3("lightColor", 1.f, 1.f, 1.f);

        groundShader->use();
        groundShader->setUniformBlockBinding("cameraUBO", 0);

        // Ground
        physics.addDefaultGround();
        auto* gnd = getScene()->definePrim("/ground", Scene::PrimType::Mesh);
        gnd->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(10.f, UpAxis::Z)));
        addShape(groundShader.get(), gnd);

        // Load H1 visual (Robot handles Prim creation + STL mesh upload)
        const std::string mjcfPath =
            KE::getAssetPath("external/retargetted/unitree_h1/unitree_h1.xml");
        robot = SkelMesh::fromMJCF(mjcfPath, getScene());
        for (auto* prim : robot.bodyPrims())
            addShape(stlShader.get(), prim);

        targets.assign(robot.numBodies() - 1, 0.f);

        artic = Articulation::build(physics, robot, mjcfPath,
                                    ArticulationConfig::fixedBase());
        checkError();
    }

    // -----------------------------------------------------------------------
    void preRender() override {
        // Spacebar: toggle pause
        bool spaceDown = glfwGetKey(getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasDown)
            paused = !paused;
        spaceWasDown = spaceDown;

        if (!paused) {
            artic.setDriveTargets(targets, kp, kd);
            physics.step();
            physics.syncAllVisuals();
        }

        // Lighting
        auto view = getViewMatrix();
        glm::vec3 lv = glm::vec3(view * glm::vec4(lightPos, 1.f));
        stlShader->use();
        stlShader->setVec3("lightPos", lv);
        stlShader->setVec3("lightColor", 1.f, 1.f, 1.f);

        groundShader->use();
        groundShader->setVec3("lightPos", lv);
        groundShader->setVec3("lightColor", 1.f, 1.f, 1.f);

        checkError();
    }

    // -----------------------------------------------------------------------
    void render() override {
        auto& tree = robot.skeleton();
        int n = tree.numJoints();

        ImGui::Begin("H1 PD Control");
        ImGui::Text("Bodies: %d  |  %s", n, paused ? "PAUSED" : "running");
        ImGui::Text("Space: pause/resume");
        ImGui::Separator();

        // PD gains
        bool gainsChanged = false;
        gainsChanged |= ImGui::SliderFloat("kp (stiffness)", &kp, 0.f, 1000.f);
        gainsChanged |= ImGui::SliderFloat("kd (damping)", &kd, 0.f, 100.f);
        ImGui::Separator();

        // Per-joint target sliders (skip root at index 0)
        ImGui::Text("Joint targets (rad):");
        ImGui::BeginChild("joints", ImVec2(0, 400), true);
        for (int i = 1; i < n; i++) {
            const auto& jd = artic.joints()[i];
            std::string label = jd.name.empty() ? tree.nodeName(i) : jd.name;
            ImGui::SliderFloat(label.c_str(), &targets[i - 1], jd.loLimit,
                               jd.hiLimit);
        }
        ImGui::EndChild();

        if (ImGui::Button("Reset to zero")) {
            std::fill(targets.begin(), targets.end(), 0.f);
        }
        ImGui::SameLine();
        if (ImGui::Button("Stand pose")) {
            // Slight knee bend for a stable stand
            std::fill(targets.begin(), targets.end(), 0.f);
            for (int i = 1; i < n; i++) {
                const std::string& nm = artic.joints()[i].name;
                if (nm == "left_knee" || nm == "right_knee")
                    targets[i - 1] = 0.4f;
                if (nm == "left_hip_pitch" || nm == "right_hip_pitch")
                    targets[i - 1] = -0.2f;
                if (nm == "left_ankle" || nm == "right_ankle")
                    targets[i - 1] = -0.2f;
            }
        }

        ImGui::End();
    }

    ~H1PhysicsApp() override = default;
};

// ---------------------------------------------------------------------------
int main() {
    H1PhysicsApp app;
    app.initialize(1920, 1080, false, UpAxis::Z);
    app.start();
    return 0;
}
