///
/// H1 Robot — Ragdoll (floating base) + Active PD Control
///
/// Identical to physx_h1.cpp except the articulation root is NOT fixed:
///   artic->setArticulationFlag(PxArticulationFlag::eFIX_BASE, false)
///
/// The robot spawns above ground and falls freely.
/// With kp=0, kd=small → pure ragdoll (limp, gravity only).
/// With kp>0               → active ragdoll (tries to hold pose while falling).
///
/// Controls:
///   Space : pause / resume
///   R     : reset robot at initial height
///   WASD + mouse : camera
///   ImGui : PD gains, per-joint target angles, spawn height
///

#include "animation/mjcf_loader.hpp"
#include "animation/robot.hpp"
#include "foundation/Px.h"
#include "kangEngine.hpp"
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
// Shaders
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
uniform vec3 camPos;
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

class H1RagdollApp : public App {
  public:
    std::unique_ptr<Backend::Shader> stlShader;
    std::unique_ptr<Backend::Shader> groundShader;

    glm::vec3 lightPos = {0.f, -2.f, 5.f};

    SkelMesh robot;
    PhysicsWorld physics{PhysicsConfig::zUp()};
    Articulation artic;

    std::vector<float> targets;

    float spawnHeightOffset = 1.5f;

    float kp = 0.f;
    float kd = 5.f;
    bool paused = false;
    bool spaceWasDown = false;
    bool rWasDown = false;
    bool showCollision = false;
    // -----------------------------------------------------------------------
    void setup() override {
        stlShader = getGraphicsDevice()->createShader(stlVs, stlFs);
        groundShader = getGraphicsDevice()->createShader(groundVs, groundFs);
        groundShader->use();
        groundShader->setUniformBlockBinding("cameraUBO", 0);
        stlShader->use();
        stlShader->setUniformBlockBinding("cameraUBO", 0);
        stlShader->setVec3("lightColor", 1.f, 1.f, 1.f);

        std::string basePath = "textures/skybox";
        setSkybox(KE::getAssetPath(basePath + "/Cubemap_Sky_08-512x512.png"),
                  /*{
                    KE::getAssetPath(basePath + "/right.jpg"),
                    KE::getAssetPath(basePath + "/left.jpg"),
                    KE::getAssetPath(basePath + "/top.jpg"),
                    KE::getAssetPath(basePath + "/bottom.jpg"),
                    KE::getAssetPath(basePath + "/front.jpg"),
                    KE::getAssetPath(basePath + "/back.jpg"),
                  },*/
                  UpAxis::Z);

        physics.addDefaultGround();
        auto* gnd = getScene()->definePrim("/ground", Scene::PrimType::Mesh);
        gnd->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(10.f, UpAxis::Z)));
        addShape(groundShader.get(), gnd);

        const std::string mjcfPath =
            KE::getAssetPath("external/retargetted/unitree_h1/unitree_h1.xml");
        robot = SkelMesh::fromMJCF(mjcfPath, getScene());
        for (auto* prim : robot.bodyPrims())
            addShape(stlShader.get(), prim);

        targets.assign(robot.numBodies() - 1, 0.f);

        artic = Articulation::build(physics, robot, mjcfPath,
                                    ArticulationConfig::freeBase());

        auto colPrims = artic.buildCollisionVisuals(getScene());
        for (auto* p : colPrims)
            addShape(stlShader.get(), p);

        reset();
        checkError();
    }

    void reset() {
        artic.resetRoot(PxTransform(PxVec3(0.f, 0.f, spawnHeightOffset),
                                    PxQuat(0.0f, 0.0f, 0.0f, 1.0f)));
    }

    // -----------------------------------------------------------------------
    void preRender() override {
        bool spaceDown = glfwGetKey(getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasDown)
            paused = !paused;
        spaceWasDown = spaceDown;

        bool rDown = glfwGetKey(getWindow(), GLFW_KEY_R) == GLFW_PRESS;
        if (rDown && !rWasDown)
            reset();
        rWasDown = rDown;

        if (!paused) {
            artic.setDriveTargets(targets, kp, kd);
            physics.step();
            physics.syncAllVisuals();
            artic.syncCollisionVisuals();
        }

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

        ImGui::Begin("H1 Ragdoll");
        ImGui::Text("Bodies: %d  |  %s", n, paused ? "PAUSED" : "running");
        ImGui::Text("Space: pause/resume    R: reset");
        ImGui::Separator();

        ImGui::Text("Mode: %s",
                    kp < 1.f ? "Pure ragdoll (kp=0)" : "Active ragdoll (kp>0)");
        ImGui::SliderFloat("kp (stiffness — 0=limp)", &kp, 0.f, 500.f);
        ImGui::SliderFloat("kd (damping)", &kd, 0.f, 50.f);
        ImGui::SliderFloat("Spawn height offset (m)", &spawnHeightOffset, 0.5f,
                           5.f);
        ImGui::Separator();

        if (ImGui::Button("Reset targets to zero"))
            std::fill(targets.begin(), targets.end(), 0.f);
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
            reset();
        ImGui::SameLine();
        if (ImGui::Checkbox("Show collision", &showCollision)) {
            artic.setCollisionVisible(showCollision);
            float alpha = showCollision ? 0.12f : 1.0f;
            for (auto* p : robot.bodyPrims()) {
                auto col = p->getDisplayColorAlpha().value_or(
                    glm::vec4(0.15f, 0.15f, 0.15f, 1.f));
                col.a = alpha;
                p->setDisplayColorAlpha(col);
            }
        }

        if (kp >= 1.f) {
            ImGui::Text("Joint targets (rad):");
            ImGui::BeginChild("joints", ImVec2(0, 350), true);
            for (int i = 1; i < n; i++) {
                const auto& jd = artic.joints()[i];
                std::string label =
                    jd.name.empty() ? tree.nodeName(i) : jd.name;
                ImGui::SliderFloat(label.c_str(), &targets[i - 1], jd.loLimit,
                                   jd.hiLimit);
            }
            ImGui::EndChild();
        } else {
            ImGui::TextDisabled("(Set kp > 0 to enable joint targets)");
        }
        ImGui::End();

        ImGui::Begin("Visualize Torque");
        ImGui::Text("Real-time Joint Torques (Nm):");
        ImGui::Separator();

        PxArticulationCache* cache = artic.raw()->createCache();
        artic.raw()->copyInternalStateToCache(
            *cache, PxArticulationCacheFlag::eJOINT_SOLVER_FORCES |
                        PxArticulationCacheFlag::eFORCE |
                        PxArticulationCacheFlag::ePOSITION |
                        PxArticulationCacheFlag::eVELOCITY);

        ImGui::BeginChild("torques", ImVec2(0, 300), true);
        for (int i = 1; i < n; i++) {
            const auto& jd = artic.joints()[i];
            std::string label = jd.name.empty() ? tree.nodeName(i) : jd.name;

            // float torque = cache->jointSolverForces[i - 1];
            // float torque = cache->jointForce[i - 1];
            float currentPos = cache->jointPosition[i - 1];
            float currentVel = cache->jointVelocity[i - 1];

            // Is this same with physx's implicit pd control?
            float targetVel = 0.0f; // for fix pose
            float torque = kp * (targets[i - 1] - currentPos) +
                           kd * (targetVel - currentVel);

            float bar = (torque + 100.f) / 200.f;
            bar = std::clamp(bar, 0.f, 1.f);
            char overlay[64];
            snprintf(overlay, sizeof(overlay), "%+.1f Nm", torque);
            ImGui::Text("%s", label.c_str());
            ImGui::SameLine();
            ImGui::ProgressBar(bar, ImVec2(-1, 0), overlay);
        }
        cache->release();
        ImGui::EndChild();
        ImGui::End();
    }
};

// ---------------------------------------------------------------------------
int main() {
    H1RagdollApp app;
    app.initialize(1920, 1080, false, UpAxis::Z);
    app.start();
    return 0;
}
