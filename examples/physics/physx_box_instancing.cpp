// #include "bridge/physics_bridge.hpp"
#include "foundation/Px.h"
#include "kangEngine.hpp"
#include "physics/physics.hpp"
#include <Eigen/Core>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <memory>
#include <vector>

using namespace KE;
using namespace physx;

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

class BoxInstancingApp : public App {
  public:
    std::unique_ptr<Backend::Shader> commonShader;
    std::unique_ptr<Backend::Shader> groundShader;

    PhysicsWorld physics{PhysicsConfig::zUp()};

    static constexpr int NUM_BOXES = 10000;
    static constexpr float BOX_HALF = 0.15f; // 30cm box
    static constexpr float SPAWN_HEIGHT_BASE = 2.0f;
    static constexpr float SPAWN_HEIGHT_STEP = 0.4f;

    std::vector<PxRigidDynamic*> _actors;
    std::vector<Scene::Prim*> _prims;
    std::shared_ptr<Scene::MeshData> _boxMesh;
    Eigen::MatrixXf _spawnPositions;
    // Bridge::PhysicsBridge physicsBridge;

    MeshHandle _boxHandle = InvalidHandle;
    std::vector<glm::mat4> _transforms; // 매 프레임 VBO 직접 업로드
    std::vector<glm::vec4> _colors;

    bool paused = false;
    bool spaceWasDown = false;
    bool rWasDown = false;

    // ---------------------------------------------------------------------------
    void setup() override {
        this->getCamera().setCameraPos(glm::vec3(10.f, 0.f, 5.f));
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

        std::string basePath = "external/skybox";
        setSkybox(KE::getAssetPath(basePath + "/Cubemap_Sky_08-512x512.png"));

        physics.addDefaultGround();

        auto* gnd = getScene()->definePrim("/ground", Scene::PrimType::Mesh);
        gnd->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(100.f, UpAxis::Z)));
        addShape(groundShader.get(), gnd);

        //_boxMesh =
        //    std::make_shared<Scene::MeshData>(Scene::Prim::createRectangleData(
        //        BOX_HALF * 2, BOX_HALF * 2, BOX_HALF * 2));
        _boxMesh = std::make_shared<Scene::MeshData>(
            Scene::Prim::createSphereData(BOX_HALF, 32, 16));
        auto* boxMeshAsset =
            getScene()->definePrim("/mesh_assets/box", Scene::PrimType::Mesh);
        boxMeshAsset->setMeshData(_boxMesh);

        _spawnPositions.resize(NUM_BOXES, 3);
        const int cols = 10;
        const int halfCols = cols / 2;
        for (int i = 0; i < NUM_BOXES; i++) {
            int col = i % cols;
            int row = i / cols;
            float x =
                static_cast<float>(col - halfCols) * (BOX_HALF * 2 + 0.05f);
            float y = static_cast<float>(row % cols - halfCols) *
                      (BOX_HALF * 2 + 0.05f);
            float z = SPAWN_HEIGHT_BASE +
                      static_cast<float>(row / cols) * SPAWN_HEIGHT_STEP;
            _spawnPositions.row(i) << x, y, z;
        }

        spawnBoxes();
        checkError();
    }

    void spawnBoxes() {
        for (int i = 0; i < NUM_BOXES; i++) {
            float x = _spawnPositions(i, 0);
            float y = _spawnPositions(i, 1);
            float z = _spawnPositions(i, 2);

            // PxRigidDynamic* actor = physics.createDynamicBox(
            //     glm::vec3(BOX_HALF), glm::vec3(x, y, z));
            PxRigidDynamic* actor = physics.createDynamicSphere(
                BOX_HALF, glm::vec3(x, y, z), glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                1.0f);
            _actors.push_back(actor);

            auto* prim = getScene()->definePrim("/boxes/b" + std::to_string(i),
                                                Scene::PrimType::MeshInstance);
            prim->setMeshSourcePath("/mesh_assets/box");
            float t = static_cast<float>(i) / NUM_BOXES;
            glm::vec4 color = glm::vec4(0.3f + 0.7f * t, 0.6f - 0.4f * t,
                                        1.0f - 0.6f * t, 1.0f);
            prim->setDisplayColorAlpha(color);
            _colors.push_back(color);
            MeshHandle h =
                addShape(commonShader.get(), prim, RenderTrack::Instanced);
            if (_boxHandle == InvalidHandle)
                _boxHandle = h;
            _prims.push_back(prim);
        }
        _transforms.resize(NUM_BOXES);
        setShapeColors(_boxHandle, _colors);
    }

    void resetBoxes() {
        for (int i = 0; i < NUM_BOXES; i++) {
            float x = _spawnPositions(i, 0);
            float y = _spawnPositions(i, 1);
            float z = _spawnPositions(i, 2);
            _actors[i]->setGlobalPose(PxTransform(PxVec3(x, y, z)));
            _actors[i]->setLinearVelocity(PxVec3(0, 0, 0));
            _actors[i]->setAngularVelocity(PxVec3(0, 0, 0));
            _actors[i]->wakeUp();
        }
    }

    // ---------------------------------------------------------------------------
    void preRender() override {
        bool spaceDown = glfwGetKey(getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasDown) {
            paused = !paused;
            if (paused)
                syncVisuals(); // sync with scene data
        }
        spaceWasDown = spaceDown;

        bool rDown = glfwGetKey(getWindow(), GLFW_KEY_R) == GLFW_PRESS;
        if (rDown && !rWasDown)
            resetBoxes();
        rWasDown = rDown;

        if (!paused) {
            physics.step();
            // bypass
            // physicsBridge.syncAllVisuals();
            // direct pass
            for (int i = 0; i < NUM_BOXES; i++)
                _transforms[i] = pxToMat4(_actors[i]->getGlobalPose());
            updateShapeTransforms(_boxHandle, _transforms);
        }
        checkError();
    }

    void syncVisuals() {
        for (int i = 0; i < NUM_BOXES; i++) {
            PxTransform pose = _actors[i]->getGlobalPose();
            _prims[i]->setAttribute("xformOp:translate", pxToGlm(pose.p));
            _prims[i]->setAttribute("xformOp:rotateQuaternion",
                                    pxToGlm(pose.q));
        }
    }

    // ---------------------------------------------------------------------------
    void render() override {
        ImGui::Begin("Box Instancing");
        ImGui::Text("Boxes: %d  |  %s", NUM_BOXES,
                    paused ? "PAUSED" : "running");
        ImGui::Text("Space: pause/resume    R: reset");
        ImGui::Separator();
        ImGui::Text("Box size: %.0f cm", BOX_HALF * 200.f);
        ImGui::End();
    }
};

// ---------------------------------------------------------------------------
int main() {
    BoxInstancingApp app;
    app.initialize(1920, 1080, false, UpAxis::Z);
    app.start();
    return 0;
}
