///
/// CPU PBD Cloth Simulation
///
/// A NxM particle grid hangs from its top two corners.
///
/// Controls:
///   Space : pause / resume
///   R     : reset cloth
///   WASD + mouse : camera
///   ImGui : gravity, stiffness, damping, iterations
///

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "kangEngine.hpp"
#include "sim/cloth/cloth_model.hpp"
#include "sim/cloth/cloth_solver.hpp"
#include "sim/cloth/cloth_state.hpp"
#include <glm/glm.hpp>
#include <imgui.h>
#include <memory>

using namespace KE;
using namespace KE::Sim;

bool isXPBD = true;

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

class PBDClothApp : public App {
  public:
    std::unique_ptr<Backend::Shader> clothShader;
    std::unique_ptr<Backend::Shader> groundShader;
    std::unique_ptr<Backend::Texture> texture = nullptr;

    ClothModel model;
    ClothState state;
    std::unique_ptr<ClothSolver> solver;
    ClothSolverConfig cfg;

    MeshHandle clothHandle = InvalidHandle;
    Scene::Prim* clothPrim = nullptr;

    bool paused = false;
    bool spaceWasDown = false;
    bool rWasDown = false;

    std::string defaultPath = "./build/assets/";
    static constexpr int ROWS = 30;
    static constexpr int COLS = 30;
    static constexpr float WIDTH = 1.2f;
    static constexpr float HEIGHT = 1.6f;
    glm::vec3 CLOTH_ORIGIN = glm::vec3(-WIDTH / 2.f, 0.f, 0.5f);

    // -----------------------------------------------------------------------
    void setup() override {
        this->getCamera().setCameraPos(glm::vec3(3.f, 3.f, 1.f));
        auto commonVSPath = KE::getAssetPath("shaders/common.vs");
        auto commonFSPath = KE::getAssetPath("shaders/common.fs");
        auto commonTexFSPath = KE::getAssetPath("shaders/commonTex.fs");
        auto groundFSPath = KE::getAssetPath("shaders/checkerboard.fs");
        clothShader = getGraphicsDevice()->createShaderFromFile(
            commonVSPath, commonTexFSPath);
        clothShader->use();
        clothShader->setUniformBlockBinding("cameraUBO", 0);
        clothShader->setUniformBlockBinding("lightUBO", 1);

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

        setSkybox(
            KE::getAssetPath("external/skybox/Cubemap_Sky_08-512x512.png"));

        auto* gnd = getScene()->definePrim("/ground", Scene::PrimType::Mesh);
        gnd->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(20.f, UpAxis::Z)));
        addShape(groundShader.get(), gnd);

        texture = getGraphicsDevice()->createTexture(
            KE::getAssetPath("textures/awesomeface.png"));
        clothShader->setInt("uTexture", 0);

        if (isXPBD) {
            solver = std::make_unique<ClothXPBDSolver>();
            cfg.stretchStiffness = 1000000.f;
            cfg.shearStiffness = 800000.f;
            cfg.bendStiffness = 1000.f;
        } else {
            solver = std::make_unique<ClothSolver>();
        }

        // Build cloth topology
        model = ClothModel::createGrid(ROWS, COLS, WIDTH, HEIGHT);

        state.reset(model, CLOTH_ORIGIN, {1, 0, 0}, {0, 1, 0});
        // state.reset(model, CLOTH_ORIGIN);
        //  Top
        state.pin(model.index(ROWS - 1, 0));
        state.pin(model.index(ROWS - 1, COLS - 1));
        // Bottom
        state.pin(model.index(0, 0));
        state.pin(model.index(0, COLS - 1));

        auto normals = ClothSolver::computeNormals(model, state);
        Scene::MeshData meshData;
        meshData.vertices = state.pos;
        meshData.normals = normals;
        meshData.uvs = model.uv;
        meshData.indices.assign(model.indices.begin(), model.indices.end());

        clothPrim = getScene()->definePrim("/cloth", Scene::PrimType::Mesh);
        clothPrim->setMeshData(std::make_shared<Scene::MeshData>(meshData));
        clothPrim->setDisplayColorAlpha(glm::vec4(0.3f, 0.6f, 1.0f, 1.0f));

        clothHandle = addShape(clothShader.get(), clothPrim);
        setShapeDoubleSided(clothHandle);
        if (texture != nullptr)
            setShapeTexture(clothHandle, texture.get(), 0);

        checkError();
    }

    // -----------------------------------------------------------------------
    void preRender() override {
        bool spaceDown = glfwGetKey(getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasDown)
            paused = !paused;
        spaceWasDown = spaceDown;

        bool rDown = glfwGetKey(getWindow(), GLFW_KEY_R) == GLFW_PRESS;
        if (rDown && !rWasDown) {
            // state.reset(model, CLOTH_ORIGIN);
            state.reset(model, CLOTH_ORIGIN, {1, 0, 0}, {0, 1, 0});
            state.pin(model.index(ROWS - 1, 0));
            state.pin(model.index(ROWS - 1, COLS - 1));
            state.pin(model.index(0, 0));
            state.pin(model.index(0, COLS - 1));
        }
        rWasDown = rDown;

        if (!paused) {
            solver->step(model, state, 1.f / 120.f, cfg);
        }

        auto normals = ClothSolver::computeNormals(model, state);
        updateMeshGeometry(clothHandle, state.pos, normals);

        checkError();
    }

    // -----------------------------------------------------------------------
    void render() override {
        if (!isXPBD) {
            ImGui::Begin("PBD Cloth");
        } else {
            ImGui::Begin("XPBD Cloth");
        }
        ImGui::Text("Particles: %d  |  %s", model.numParticles(),
                    paused ? "PAUSED" : "running");
        ImGui::Text("Space: pause/resume    R: reset");
        ImGui::Separator();

        ImGui::SliderFloat3("Gravity", &cfg.gravity.x, -20.f, 20.f);
        ImGui::SliderFloat("Damping", &cfg.damping, 0.9f, 1.0f);
        // ImGui::SliderInt("Iterations", &cfg.iterations, 1, 32);
        ImGui::SliderInt("Num substeps", &cfg.numSubsteps, 1, 128);
        ImGui::Separator();
        if (!isXPBD) {
            ImGui::SliderFloat("Stretch stiffness", &cfg.stretchStiffness, 0.f,
                               1.f);
            ImGui::SliderFloat("Shear stiffness", &cfg.shearStiffness, 0.f,
                               1.f);
            ImGui::SliderFloat("Bend stiffness", &cfg.bendStiffness, 0.f, 1.f);
        } else {
            ImGui::SliderFloat("Stretch stiffness", &cfg.stretchStiffness, 0.f,
                               1000000.f);
            ImGui::SliderFloat("Shear stiffness", &cfg.shearStiffness, 0.f,
                               800000.f);
            ImGui::SliderFloat("Bend stiffness", &cfg.bendStiffness, 0.f,
                               300.f);
        }
        ImGui::End();
    }
};

// ---------------------------------------------------------------------------
int main() {
    PBDClothApp app;
    app.initialize(1920, 1080, false, UpAxis::Z);
    app.start();
    return 0;
}
