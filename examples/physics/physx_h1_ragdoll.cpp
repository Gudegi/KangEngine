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

#include "kangEngine.hpp"
#include "foundation/Px.h"
#include <Eigen/Geometry>
#include <glm/glm.hpp>
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>

using namespace KE;
using namespace KE::Animation;
using namespace KE::Bridge;
using namespace physx;

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

class H1RagdollApp : public App {
  public:
    std::unique_ptr<Backend::Shader> commonShader;
    std::unique_ptr<Backend::Shader> groundShader;

    SkeletonBridge robot;
    PhysicsWorld physics{PhysicsConfig::zUp()};
    Articulation artic;
    Bridge::PhysicsBridge physicsBridge;

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
        auto commonVSPath = KE::getAssetPath("shaders/common.vs");
        auto commonFSPath = KE::getAssetPath("shaders/common.fs");
        auto groundFSPath = KE::getAssetPath("shaders/checkerboard.fs");

        commonShader = getGraphicsDevice()->createShaderFromFile(commonVSPath,
                                                                 commonFSPath);
        groundShader = getGraphicsDevice()->createShaderFromFile(commonVSPath,
                                                                 groundFSPath);

        groundShader->use();
        groundShader->setUniformBlockBinding("cameraUBO", 0);
        groundShader->setUniformBlockBinding("lightUBO", 1);
        auto white = ColorLibrary::get(KE::ColorType::WHITE);
        auto pG = ColorLibrary::get(KE::ColorType::PASTEL_GREEN);
        groundShader->setVec4("checkerColor1",
                              glm::vec4(white.r, white.g, white.b, white.a));
        groundShader->setVec4("checkerColor2",
                              glm::vec4(pG.r, pG.g, pG.b, pG.a));
        commonShader->use();
        commonShader->setUniformBlockBinding("cameraUBO", 0);
        commonShader->setUniformBlockBinding("lightUBO", 1);

        std::string basePath = "external/skybox";
        setSkybox(KE::getAssetPath(basePath + "/Cubemap_Sky_08-512x512.png")
                  /*{
                    KE::getAssetPath(basePath + "/right.jpg"),
                    KE::getAssetPath(basePath + "/left.jpg"),
                    KE::getAssetPath(basePath + "/top.jpg"),
                    KE::getAssetPath(basePath + "/bottom.jpg"),
                    KE::getAssetPath(basePath + "/front.jpg"),
                    KE::getAssetPath(basePath + "/back.jpg"),
                  },*/);

        physics.addDefaultGround();
        auto* gnd = getScene()->definePrim("/ground", Scene::PrimType::Mesh);
        gnd->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(100.f, UpAxis::Z)));
        addShape(groundShader.get(), gnd);

        // const std::string mjcfPath =
        //     KE::getAssetPath("external/retargetted/unitree_h1/unitree_h1.xml");
        const std::string mjcfPath =
            KE::getAssetPath("external/retargetted/kw/kw5.xml");
        const auto mjcfData = MJCFLoader::load(mjcfPath);

        artic = Articulation::build(physics, mjcfData.skeletonTree,
                                    mjcfData.collisionGeoms, mjcfData.joints,
                                    mjcfData.inertials,
                                    ArticulationConfig::freeBase());

        robot = SkeletonBridge::fromData(mjcfData, getScene());
        physicsBridge.add(artic, robot);
        for (auto* prim : robot.bodyPrims())
            addShape(commonShader.get(), prim);

        auto colPrims = physicsBridge.addCollisionVisuals(artic, getScene());
        for (auto* p : colPrims)
            addShape(commonShader.get(), p);

        targets.assign(artic.numDofs(), 0.f);

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
            physicsBridge.sync();
        }

        checkError();
    }

    // -----------------------------------------------------------------------
    void render() override {
        int n = artic.numLinks();

        ImGui::Begin("H1 Ragdoll");
        ImGui::Text("Bodies: %d  DOFs: %d  |  %s", n, artic.numDofs(),
                    paused ? "PAUSED" : "running");
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
            physicsBridge.setCollisionVisible(showCollision);
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
            int dofIdx = 0;
            for (int i = 1; i < n; i++) {
                auto jit = artic.joints().find(i);
                if (jit == artic.joints().end())
                    continue;
                for (const auto& jd : jit->second)
                    ImGui::SliderFloat(jd.name.c_str(), &targets[dofIdx++],
                                       jd.loLimit, jd.hiLimit);
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
        {
            int dofIdx = 0;
            for (int i = 1; i < n; i++) {
                auto jit = artic.joints().find(i);
                if (jit == artic.joints().end())
                    continue;
                for (const auto& jd : jit->second) {
                    // Is this same with physx's implicit pd control?
                    float currentPos = cache->jointPosition[dofIdx];
                    float currentVel = cache->jointVelocity[dofIdx];
                    float torque = kp * (targets[dofIdx] - currentPos) +
                                   kd * (0.f - currentVel);
                    float bar = (torque + 100.f) / 200.f;
                    bar = std::clamp(bar, 0.f, 1.f);
                    char overlay[64];
                    snprintf(overlay, sizeof(overlay), "%+.1f Nm", torque);
                    ImGui::Text("%s", jd.name.c_str());
                    ImGui::SameLine();
                    ImGui::ProgressBar(bar, ImVec2(-1, 0), overlay);
                    dofIdx++;
                }
            }
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
