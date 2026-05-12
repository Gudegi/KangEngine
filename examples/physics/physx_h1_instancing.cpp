///
/// H1 Robot — N-robot instanced rendering with PhysicsBridge::syncInstanced()
///
/// N identical H1 robots in a grid, all controlled by one shared PD target
/// array. Rendering uses one instancer per body type (M body types x N robots).
///
/// Controls:
///   Space : pause / resume
///   R     : reset all robots
///   WASD + mouse : camera
///

#include "kangEngine.hpp"
#include <Eigen/Geometry>
#include <glm/glm.hpp>
#include <imgui.h>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace KE;
using namespace KE::Animation;
using namespace KE::Bridge;
using namespace physx;

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

class H1InstancingApp : public App {
  public:
    static constexpr int N = 100;
    static constexpr int COLS = 5;
    static constexpr float SPACING = 1.8f;
    static constexpr float SPAWN_HEIGHT = 1.5f;

    std::unique_ptr<Backend::Shader> commonShader;
    std::unique_ptr<Backend::Shader> groundShader;

    PhysicsWorld physics{PhysicsConfig::zUp()};
    SkeletonBridge refRobot; // MeshInstance prims used for instancer setup
    std::vector<Articulation> _artics;
    Bridge::PhysicsBridge physicsBridge;

    std::vector<MeshHandle> _bodyHandles;
    std::vector<std::vector<float>> _targets; // per-robot random targets

    float kp = 200.f;
    float kd = 20.f;
    bool paused = false;
    bool spaceWasDown = false;
    bool rWasDown = false;
    std::mt19937 rng{42};

    // -----------------------------------------------------------------------
    void setup() override {
        auto commonVSPath = KE::getAssetPath("shaders/common.vs");
        auto commonFSPath = KE::getAssetPath("shaders/common.fs");
        auto groundFSPath = KE::getAssetPath("shaders/checkerboard.fs");

        commonShader = getGraphicsDevice()->createShaderFromFile(commonVSPath,
                                                                 commonFSPath);
        groundShader = getGraphicsDevice()->createShaderFromFile(commonVSPath,
                                                                 groundFSPath);

        commonShader->use();
        commonShader->setUniformBlockBinding("cameraUBO", 0);
        commonShader->setUniformBlockBinding("lightUBO", 1);

        groundShader->use();
        groundShader->setUniformBlockBinding("cameraUBO", 0);
        groundShader->setUniformBlockBinding("lightUBO", 1);
        auto white = ColorLibrary::get(KE::ColorType::WHITE);
        auto pG = ColorLibrary::get(KE::ColorType::PASTEL_GREEN);
        groundShader->setVec4("checkerColor1",
                              glm::vec4(white.r, white.g, white.b, white.a));
        groundShader->setVec4("checkerColor2",
                              glm::vec4(pG.r, pG.g, pG.b, pG.a));

        setSkybox(
            KE::getAssetPath("external/skybox/Cubemap_Sky_08-512x512.png"));

        physics.addDefaultGround();
        auto* gnd = getScene()->definePrim("/ground", Scene::PrimType::Mesh);
        gnd->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(100.f, UpAxis::Z)));
        addShape(groundShader.get(), gnd);

        // Load MJCF once.
        const std::string mjcfPath =
            KE::getAssetPath("external/retargetted/unitree_h1/unitree_h1.xml");
        const auto mjcfData = MJCFLoader::load(mjcfPath);

        // Old pattern: create direct Mesh prims as a renderer setup reference.
        // This still works, but every reference prim owns mesh data directly.
        // refRobot = SkeletonBridge::fromData(mjcfData, getScene());

        // Current pattern: create source Mesh prims once, then use
        // MeshInstance body prims as the renderer setup reference.
        auto refAsset = SkeletonBridgeAsset::fromData(mjcfData);
        const std::string meshAssetBasePath = "/mesh_assets/h1";
        refAsset.defineMeshAssets(getScene(), meshAssetBasePath);
        refRobot =
            refAsset.instantiate(getScene(), "/h1_ref", meshAssetBasePath);

        // One handle per body type — each instancer will hold N transforms
        for (auto* prim : refRobot.bodyPrims())
            _bodyHandles.push_back(
                addShape(commonShader.get(), prim, RenderTrack::Instanced));

        // Per-robot colors — same color for all bodies of one robot
        std::vector<glm::vec4> colors(N);
        for (int i = 0; i < N; i++) {
            float t = static_cast<float>(i) / N;
            colors[i] = glm::vec4(0.3f + 0.7f * t, 0.8f - 0.5f * t,
                                  0.4f + 0.4f * t, 1.0f);
        }
        for (auto h : _bodyHandles)
            setShapeColors(h, colors);

        // Build N articulations
        _artics.resize(N);
        std::vector<Articulation*> articPtrs(N);
        for (int i = 0; i < N; i++) {
            _artics[i] = Articulation::build(
                physics, mjcfData.skeletonTree, mjcfData.collisionGeoms,
                mjcfData.joints, mjcfData.inertials,
                ArticulationConfig::freeBase());
            articPtrs[i] = &_artics[i];
        }

        physicsBridge = Bridge::PhysicsBridge{this};
        physicsBridge.addInstanced(articPtrs, _bodyHandles);

        // random targets per robot, clamped to joint limits
        _targets.resize(N);
        for (int i = 0; i < N; i++) {
            int numDofs = _artics[i].numDofs();
            _targets[i].resize(numDofs);
            int dofIdx = 0;
            for (int b = 1; b < _artics[i].numLinks(); b++) {
                auto jit = _artics[i].joints().find(b);
                if (jit == _artics[i].joints().end())
                    continue;
                for (const auto& jd : jit->second) {
                    std::uniform_real_distribution<float> dist(jd.loLimit,
                                                               jd.hiLimit);
                    _targets[i][dofIdx++] = dist(rng);
                }
            }
        }

        spawnRobots();
        checkError();
    }

    void spawnRobots() {
        for (int i = 0; i < N; i++) {
            int col = i % COLS;
            int row = i / COLS;
            float x = (col - COLS / 2) * SPACING;
            float y = (row - 1) * SPACING;
            _artics[i].resetRoot(PxTransform(PxVec3(x, y, SPAWN_HEIGHT),
                                             PxQuat(0.f, 0.f, 0.f, 1.f)));
        }
    }

    // -----------------------------------------------------------------------
    void preRender() override {
        bool spaceDown = glfwGetKey(getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasDown)
            paused = !paused;
        spaceWasDown = spaceDown;

        bool rDown = glfwGetKey(getWindow(), GLFW_KEY_R) == GLFW_PRESS;
        if (rDown && !rWasDown)
            spawnRobots();
        rWasDown = rDown;

        for (int i = 0; i < N; i++) {
            int dofIdx = 0;
            for (int b = 1; b < _artics[i].numLinks(); b++) {
                auto jit = _artics[i].joints().find(b);
                if (jit == _artics[i].joints().end())
                    continue;
                for (const auto& jd : jit->second) {
                    std::uniform_real_distribution<float> dist(jd.loLimit,
                                                               jd.hiLimit);
                    _targets[i][dofIdx++] = dist(rng);
                }
            }
        }

        if (!paused) {
            for (int i = 0; i < N; i++)
                _artics[i].setDriveTargets(_targets[i], kp, kd);
            physics.step();
        }
        physicsBridge.sync();

        checkError();
    }

    // -----------------------------------------------------------------------
    void render() override {
        ImGui::Begin("H1 Instancing");
        ImGui::Text("Robots: %d  |  %s", N, paused ? "PAUSED" : "running");
        ImGui::Text("Space: pause/resume    R: reset");
        ImGui::Separator();
        ImGui::SliderFloat("kp (stiffness)", &kp, 0.f, 500.f);
        ImGui::SliderFloat("kd (damping)", &kd, 0.f, 50.f);
        ImGui::End();
    }
};

// ---------------------------------------------------------------------------
int main() {
    H1InstancingApp app;
    app.initialize(1920, 1080, false, UpAxis::Z);
    app.start();
    return 0;
}
