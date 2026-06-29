#include "kangEngine.hpp"
#include "physics/physics.hpp"
#include "physics/physics_gpu_system.hpp"

using namespace KE;
using namespace physx;

namespace {

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

glm::mat4 rigidRowToMat4(const float* row) {
    const glm::vec3 pos(row[0], row[1], row[2]);
    const glm::quat rot(row[6], row[3], row[4], row[5]); // xyzw to wxyz
    return glm::translate(glm::mat4(1.0f), pos) * glm::mat4_cast(rot);
}

float noise(int index, int multiplier) {
    return static_cast<float>((index * multiplier) % 31 - 15) * 0.001f;
}

} // namespace

class GpuBoxInstancingApp : public App {
  public:
    std::unique_ptr<Backend::Shader> commonShader;
    std::unique_ptr<Backend::Shader> groundShader;

    PhysicsWorld physics{PhysicsConfig::zUp()};
    std::unique_ptr<PhysicsGpuSystem> gpuSystem;

    static constexpr int NUM_BOXES = 10000;
    static constexpr float BOX_HALF = 0.15f;
    static constexpr float SPAWN_HEIGHT_BASE = 2.0f;
    static constexpr float SPAWN_HEIGHT_STEP = 0.4f;

    std::vector<PxRigidDynamic*> _actors;
    std::shared_ptr<Scene::MeshData> _boxMesh;
    Eigen::MatrixXf _spawnPositions;
    RenderableHandle _boxHandle = InvalidHandle;
    std::vector<glm::mat4> _transforms;
    std::vector<glm::vec4> _colors;
    std::vector<float> _rigidHost;
    std::vector<float> _resetRigidHost;
    glm::vec3 _positionMin{0.0f};
    glm::vec3 _positionMax{0.0f};
    uint64_t _transformVersion = 0;

    bool paused = false;
    bool noiseEnabled = true;
    bool spaceWasDown = false;
    bool rWasDown = false;
    bool nWasDown = false;

    void setup() override {
        this->getCamera().setCameraPos(glm::vec3(10.f, 0.f, 5.f));

        commonShader = getRenderer().device()->createShaderFromFile(
            KE::getAssetPath("shaders/common.vs"),
            KE::getAssetPath("shaders/common.fs"));
        groundShader = getRenderer().device()->createShaderFromFile(
            KE::getAssetPath("shaders/common.vs"),
            KE::getAssetPath("shaders/checkerboard.fs"));

        commonShader->use();
        commonShader->setUniformBlockBinding("cameraUBO", 0);
        commonShader->setUniformBlockBinding("lightUBO", 1);
        commonShader->setUniformBlockBinding("shadowUBO", 2);

        groundShader->use();
        groundShader->setUniformBlockBinding("cameraUBO", 0);
        groundShader->setUniformBlockBinding("lightUBO", 1);
        groundShader->setUniformBlockBinding("shadowUBO", 2);
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
        addRenderable(groundShader.get(), gnd);

        _boxMesh = std::make_shared<Scene::MeshData>(
            Scene::Prim::createSphereData(BOX_HALF, 32, 16));
        auto* boxMeshAsset = getScene()->definePrim("/mesh_assets/gpu_box",
                                                    Scene::PrimType::Mesh);
        boxMeshAsset->setMeshData(_boxMesh);

        _spawnPositions.resize(NUM_BOXES, 3);
        const int cols = 10;
        const int halfCols = cols / 2;
        for (int i = 0; i < NUM_BOXES; i++) {
            const int col = i % cols;
            const int row = i / cols;
            const float x =
                static_cast<float>(col - halfCols) * (BOX_HALF * 2 + 0.05f);
            const float y = static_cast<float>(row % cols - halfCols) *
                            (BOX_HALF * 2 + 0.05f);
            const float z = SPAWN_HEIGHT_BASE +
                            static_cast<float>(row / cols) * SPAWN_HEIGHT_STEP;
            _spawnPositions.row(i) << x, y, z;
        }

        spawnBoxes();

        GpuPhysicsConfig gpuConfig;
        gpuConfig.cudaDeviceId = 0;
        gpuSystem = std::make_unique<PhysicsGpuSystem>(&physics, gpuConfig);
        gpuSystem->init();

        _rigidHost.resize(static_cast<size_t>(NUM_BOXES) * 13);
        _transforms.resize(NUM_BOXES, glm::mat4(1.0f));
        uploadGpuTransforms();
        _resetRigidHost = _rigidHost;
        updateResetRigidState();
        checkError();
    }

    glm::vec3 spawnPosition(int index) const {
        glm::vec3 position(_spawnPositions(index, 0), _spawnPositions(index, 1),
                           _spawnPositions(index, 2));
        if (noiseEnabled) {
            position.x += noise(index, 11);
            position.y += noise(index, 23);
        }
        return position;
    }

    void updateResetRigidState() {
        for (int i = 0; i < NUM_BOXES; ++i) {
            float* row = &_resetRigidHost[static_cast<size_t>(i) * 13];
            const glm::vec3 position = spawnPosition(i);
            row[0] = position.x;
            row[1] = position.y;
            row[2] = position.z;
            row[7] = 0.0f;
            row[8] = 0.0f;
            row[9] = 0.0f;
            row[10] = 0.0f;
            row[11] = 0.0f;
            row[12] = 0.0f;
        }
    }

    void spawnBoxes() {
        _actors.reserve(NUM_BOXES);
        _colors.reserve(NUM_BOXES);
        for (int i = 0; i < NUM_BOXES; i++) {
            const glm::vec3 position = spawnPosition(i);

            PxRigidDynamic* actor = physics.createDynamicSphere(
                BOX_HALF, position, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), 1.0f);
            _actors.push_back(actor);

            const float t = static_cast<float>(i) / NUM_BOXES;
            _colors.emplace_back(0.3f + 0.7f * t, 0.6f - 0.4f * t,
                                 1.0f - 0.6f * t, 1.0f);
        }

        auto* owner =
            getScene()->definePrim("/gpu_boxes", Scene::PrimType::MeshInstance);
        owner->setMeshSourcePath("/mesh_assets/gpu_box");
        owner->setDisplayColorAlpha(glm::vec4(1.0f));
        _boxHandle = addRenderable(commonShader.get(), owner,
                                   TransformSource::ExternalBuffer);
        setRenderableColors(_boxHandle, _colors);
    }

    void resetBoxes() {
        const auto& rigidView = gpuSystem->rigidData();
        checkCuda(cudaMemcpy(rigidView.data, _resetRigidHost.data(),
                             sizeof(float) * _resetRigidHost.size(),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(reset rigid state)");
        gpuSystem->applyRigidData();
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(reset)");
        uploadGpuTransforms();
    }

    void uploadGpuTransforms() {
        gpuSystem->fetchRigidData();
        const auto& rigidView = gpuSystem->rigidData();
        if (rigidView.shape.size() != 2 || rigidView.shape[0] != NUM_BOXES ||
            rigidView.shape[1] != 13)
            throw std::runtime_error("unexpected rigid GPU state shape");

        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        checkCuda(cudaMemcpy(_rigidHost.data(), rigidView.data,
                             sizeof(float) * _rigidHost.size(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(rigid state)");

        _positionMin = glm::vec3(std::numeric_limits<float>::max());
        _positionMax = glm::vec3(std::numeric_limits<float>::lowest());
        for (int i = 0; i < NUM_BOXES; ++i) {
            const float* row = &_rigidHost[static_cast<size_t>(i) * 13];
            const glm::vec3 position(row[0], row[1], row[2]);
            _positionMin = glm::min(_positionMin, position);
            _positionMax = glm::max(_positionMax, position);
            _transforms[static_cast<size_t>(i)] = rigidRowToMat4(row);
        }

        ExternalBufferDesc desc;
        desc.view.data = _transforms.data();
        desc.view.memoryType =
            Sim::SimMemoryType::CpuHost; // TODO: GPU compatible
        desc.view.dtype = Sim::SimDType::Float32;
        desc.view.lifetime = Sim::SimLifetimePolicy::ExternalOwner;
        desc.view.shape = {NUM_BOXES, 4, 4};
        desc.view.strides = {16, 4, 1};
        desc.view.version = ++_transformVersion;
        desc.view.name = "gpu_box_world_transforms";
        desc.format = ExternalBufferFormat::Mat4;
        desc.count = NUM_BOXES;
        desc.syncPolicy = ExternalSyncPolicy::Versioned;
        getRenderer().setRenderableExternalBuffer(_boxHandle, desc);
    }

    void preRender() override {
        bool spaceDown = glfwGetKey(getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasDown)
            paused = !paused;
        spaceWasDown = spaceDown;

        bool rDown = glfwGetKey(getWindow(), GLFW_KEY_R) == GLFW_PRESS;
        if (rDown && !rWasDown)
            resetBoxes();
        rWasDown = rDown;

        bool nDown = glfwGetKey(getWindow(), GLFW_KEY_N) == GLFW_PRESS;
        if (nDown && !nWasDown) {
            noiseEnabled = !noiseEnabled;
            updateResetRigidState();
            resetBoxes();
        }
        nWasDown = nDown;

        if (!paused) {
            physics.step();
            uploadGpuTransforms();
        }
        checkError();
    }

    void render() override {
        ImGui::Begin("GPU Box Instancing");
        ImGui::Text("Boxes: %d  |  %s", NUM_BOXES,
                    paused ? "PAUSED" : "running");
        ImGui::Text("Source: PhysX Direct GPU rigid buffer");
        ImGui::Text("Renderer upload: CPU mat4 external buffer");
        ImGui::Text("X range: %.2f .. %.2f", _positionMin.x, _positionMax.x);
        ImGui::Text("Y range: %.2f .. %.2f", _positionMin.y, _positionMax.y);
        ImGui::Text("Noise: %s", noiseEnabled ? "ON" : "OFF");
        ImGui::Text("Space: pause/resume    R: reset    N: toggle noise");
        ImGui::Separator();
        ImGui::Text("Box size: %.0f cm", BOX_HALF * 200.f);
        ImGui::End();
    }
};

int main() {
    GpuBoxInstancingApp app;
    app.initialize(1920, 1080, false, UpAxis::Z);
    app.start();
    return 0;
}
