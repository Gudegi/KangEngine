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
#include "physics/force_drag_controller.hpp"
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <imgui.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace KE;
using namespace KE::Asset;
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
    ForceDragController forceDrag;
    ForceDragConfig forceDragConfig;

    std::vector<float> targets;
    std::vector<RenderableHandle> bodyHandles;
    RenderableHandle contactArrowHandle = InvalidHandle;

    float spawnHeightOffset = 1.5f;
    float contactImpulseScale = 0.08f;
    float contactMaxArrowLength = 0.6f;
    float dragForceArrowScale = 0.003f;

    float kp = 0.f;
    float kd = 5.f;
    bool paused = false;
    bool spaceWasDown = false;
    bool rWasDown = false;
    bool showCollision = false;
    bool showContacts = true;
    bool showDragForceArrow = true;
    // -----------------------------------------------------------------------
    void setup() override {
        auto commonVSPath = KE::getAssetPath("shaders/common.vs");
        auto commonFSPath = KE::getAssetPath("shaders/common.fs");
        auto groundFSPath = KE::getAssetPath("shaders/checkerboard.fs");

        commonShader = getRenderer().device()->createShaderFromFile(
            commonVSPath, commonFSPath);
        groundShader = getRenderer().device()->createShaderFromFile(
            commonVSPath, groundFSPath);

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
        addRenderable(groundShader.get(), gnd);

        // const std::string mjcfPath =
        //     KE::getAssetPath("external/retargetted/unitree_h1/unitree_h1.xml");
        const std::string mjcfPath = KE::getAssetPath("characters/kw/kw5.xml");
        const auto mjcfData = MJCFLoader::load(mjcfPath);

        artic = Articulation::build(physics, mjcfData.skeletonTree,
                                    mjcfData.collisionGeoms, mjcfData.joints,
                                    mjcfData.inertials,
                                    ArticulationConfig::freeBase());

        robot = SkeletonBridge::fromData(mjcfData, getScene());
        physicsBridge.add(artic, robot);
        bodyHandles.clear();
        bodyHandles.reserve(robot.bodyPrims().size());
        for (auto* prim : robot.bodyPrims())
            bodyHandles.push_back(addRenderable(commonShader.get(), prim));
        forceDrag.registerArticulation(artic, bodyHandles);

        auto colPrims = physicsBridge.addCollisionVisuals(artic, getScene());
        for (auto* p : colPrims)
            addRenderable(commonShader.get(), p);

        contactArrowHandle = Scene::DebugDraw::logArrows(
            this, commonShader.get(), "/debug/contact_arrows",
            {glm::vec3(0.0f)}, {glm::vec3(0.0f, 0.0f, 0.1f)},
            {glm::vec4(1.0f, 0.2f, 0.05f, 1.0f)}, 0.025f, 12);

        targets.assign(artic.numDofs(), 0.f);

        reset();
        checkError();
    }

    void onForceDragBegin(const RayPickResult& result,
                          const glm::vec3& target) override {
        forceDrag.begin(result, target);
        updateDragForceArrow();
    }

    void onForceDragUpdate(const RayPickResult&,
                           const glm::vec3& target) override {
        forceDrag.update(target);
        updateDragForceArrow();
    }

    void onForceDragEnd() override {
        forceDrag.end();
        clearDebugLines("/debug/drag_force");
        clearDebugPoints("/debug/drag_force_target");
    }

    void updateDragForceArrow() {
        if (!showDragForceArrow || !forceDrag.active()) {
            clearDebugLines("/debug/drag_force");
            clearDebugPoints("/debug/drag_force_target");
            return;
        }

        const glm::vec3 start = forceDrag.lastAnchorPosition();
        const glm::vec3 force = forceDrag.lastForce();
        const float forceLen = glm::length(force);
        if (forceLen < 1e-5f) {
            clearDebugLines("/debug/drag_force");
            return;
        }

        const glm::vec3 dir = force / forceLen;
        const glm::vec3 end = start + force * dragForceArrowScale;
        const float shaftLen = glm::length(end - start);
        const float headLen = std::clamp(shaftLen * 0.25f, 0.04f, 0.18f);

        glm::vec3 side = glm::cross(dir, glm::vec3(0.0f, 0.0f, 1.0f));
        if (glm::length(side) < 1e-5f)
            side = glm::cross(dir, glm::vec3(0.0f, 1.0f, 0.0f));
        side = glm::normalize(side);

        const glm::vec3 back = end - dir * headLen;
        const glm::vec3 left = back + side * headLen * 0.45f;
        const glm::vec3 right = back - side * headLen * 0.45f;
        const auto red = ColorLibrary::get(ColorType::RED);
        const glm::vec4 arrowColor(red.r, red.g, red.b, 1.0f);
        logDebugLines("/debug/drag_force", {start, end, end},
                      {end, left, right}, {arrowColor}, 3.0f);

        const auto mag = ColorLibrary::get(ColorType::MAGENTA);
        logDebugPoints("/debug/drag_force_target", {forceDrag.lastTarget()},
                       {glm::vec4(mag.r, mag.g, mag.b, 1.0f)}, 10.0f);
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
        updateContactArrows();
        updateDragForceArrow();

        checkError();
    }

    void updateContactArrows() {
        if (contactArrowHandle == InvalidHandle)
            return;

        std::vector<glm::mat4> transforms;
        std::vector<glm::vec4> colors;
        if (showContacts) {
            const auto& contacts = physics.getContacts();
            transforms.reserve(contacts.size());
            colors.reserve(contacts.size());

            for (const auto& contact : contacts) {
                float impulseLen = glm::length(contact.impulse);
                glm::vec3 dir = impulseLen > 1e-5f
                                    ? glm::normalize(contact.impulse)
                                    : contact.normal;
                if (glm::length(dir) < 1e-5f)
                    continue;

                float arrowLen = std::clamp(impulseLen * contactImpulseScale,
                                            0.05f, contactMaxArrowLength);
                glm::vec3 start = contact.position;
                glm::vec3 end = start + dir * arrowLen;
                glm::mat4 transform(1.0f);
                if (!Scene::DebugDraw::makeArrowTransform(start, end,
                                                          transform))
                    continue;
                transforms.push_back(transform);

                float heat =
                    std::clamp(arrowLen / contactMaxArrowLength, 0.0f, 1.0f);
                colors.push_back(
                    glm::vec4(1.0f, 0.9f - 0.7f * heat, 0.05f, 1.0f));
            }
        }
        updateRenderableTransforms(contactArrowHandle, transforms, &colors);
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
        ImGui::Checkbox("Show contacts", &showContacts);
        ImGui::SliderFloat("Contact impulse scale", &contactImpulseScale,
                           0.005f, 0.3f);
        ImGui::SliderFloat("Contact max arrow length", &contactMaxArrowLength,
                           0.05f, 2.0f);
        ImGui::Text("Contacts: %u", physics.numContacts());
        ImGui::Separator();
        ImGui::Text("Force drag: select Force mode, then Shift + left drag");
        ImGui::Checkbox("Show drag force arrow", &showDragForceArrow);
        if (ImGui::SliderFloat("Drag stiffness", &forceDragConfig.stiffness,
                               0.0f, 1000.0f) ||
            ImGui::SliderFloat("Drag damping", &forceDragConfig.damping, 0.0f,
                               80.0f) ||
            ImGui::SliderFloat("Drag max force", &forceDragConfig.maxForce,
                               0.0f, 1500.0f)) {
            forceDrag.setConfig(forceDragConfig);
        }
        ImGui::SliderFloat("Drag arrow scale", &dragForceArrowScale, 0.0005f,
                           0.02f);
        ImGui::Text("Dragging: %s", forceDrag.active() ? "yes" : "no");

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
