///
/// MJCF Articulation Gallery — load multiple MJCF characters into PhysX.
///
/// With no command-line arguments, this uses the MimicKit asset set as a
/// convenient default preset. Pass one or more MJCF XML paths to inspect other
/// assets. KANGENGINE_MJCF_ASSET_ROOT can relocate the default preset.
///
/// Every character is a free-base passive articulation. Starting the simulation
/// therefore turns the lineup into a ragdoll-style gravity/collision test for
/// the MJCF loader, articulation builder, visual bridge, and collision bridge.
///
/// Controls:
///   Enter : pause / resume
///   Space : pause / single step
///   R     : reset all loaded assets
///   C     : show / hide collision debug
///

#include "kangEngine.hpp"
#include "geometry/primitive_mesh.hpp"

#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <glm/glm.hpp>
#include <imgui.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace KE;
using namespace KE::Asset;
using namespace KE::Animation;
using namespace KE::Bridge;
using namespace physx;

namespace fs = std::filesystem;

namespace {

struct AssetSpec {
    std::string label;
    std::string path;
};

struct LoadedAsset {
    AssetSpec spec;
    int articulationIndex = -1;
    int numLinks = 0;
    int numDofs = 0;
    float x = 0.0f;
    float y = 0.0f;
    float z = 1.2f;
    float tilt = 0.0f;
    glm::vec4 color = glm::vec4(1.0f);
    std::vector<Scene::Prim*> bodyPrims;
};

struct FailedAsset {
    AssetSpec spec;
    std::string reason;
};

std::string sanitizePathToken(std::string value) {
    for (char& ch : value) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_'))
            ch = '_';
    }
    return value;
}

} // namespace

class MjcfArticulationGalleryApp : public App {
  public:
    explicit MjcfArticulationGalleryApp(std::vector<AssetSpec> assets)
        : _assets(std::move(assets)) {}

    VertexColorMaterial commonMaterial;
    VertexColorMaterial groundMaterial{VertexColorStyle::Checkerboard};

    PhysicsWorld physics{PhysicsConfig::zUp()};
    PhysicsBridge physicsBridge{};

    std::vector<Articulation> _artics;
    std::vector<LoadedAsset> _loaded;
    std::vector<FailedAsset> _failed;

    bool showCollision = false;
    bool rWasDown = false;
    bool cWasDown = false;

    std::vector<AssetSpec> _assets;

    void setup() override {
        getCamera().setCameraPos(glm::vec3(4.5f, -8.0f, 3.0f));
        getRenderer().setLight(
            DirectionalLight{glm::normalize(glm::vec3(0.35f, 0.45f, 0.82f)),
                             glm::vec3(1.0f), 0.75f, glm::vec3(0.10f)});

        auto white = ColorLibrary::get(KE::ColorType::WHITE);
        auto pG = ColorLibrary::get(KE::ColorType::PASTEL_GREEN);
        getRenderer().settings().background.checkerColor1 =
            glm::vec4(white.r, white.g, white.b, white.a);
        getRenderer().settings().background.checkerColor2 =
            glm::vec4(pG.r, pG.g, pG.b, pG.a);

        setFixedUpdateHz(60.0);
        setSimulationHotkeysEnabled(true);
        setSimulationPaused(true);

        setSkybox(
            KE::getAssetPath("external/skybox/Cubemap_Sky_08-512x512.png"));

        physics.addDefaultGround();

        auto* ground = getScene()->definePrim("/ground", Scene::PrimType::Mesh);
        ground->setMeshData(std::make_shared<Scene::MeshData>(
            Geometry::createPlane(100.f, UpAxis::Z)));
        getSceneRenderSystem().addRenderable(*ground, &groundMaterial);

        _artics.reserve(_assets.size());
        loadAssets();
        resetAssets();
        physicsBridge.sync();
        checkError();
    }

    fs::path defaultAssetRoot() const {
        if (const char* value = std::getenv("KANGENGINE_MJCF_ASSET_ROOT")) {
            if (*value != '\0')
                return fs::path(value);
        }
        fs::path sourceRoot = fs::path(KANGENGINE_ASSETS_ROOT).parent_path();
        return sourceRoot / "references" / "MimicKit" / "data" / "assets";
    }

    void loadAssets() {
        const fs::path assetRoot = defaultAssetRoot();
        const int cols = 4;
        const float spacing = 2.4f;

        std::cout << "Default MJCF asset root: " << assetRoot << "\n";

        for (int i = 0; i < static_cast<int>(_assets.size()); i++) {
            const AssetSpec& spec = _assets[i];
            const fs::path specPath = spec.path;
            const fs::path xmlPath =
                specPath.is_absolute() ? specPath : assetRoot / specPath;
            const std::string token = sanitizePathToken(spec.label);

            try {
                if (!fs::exists(xmlPath))
                    throw std::runtime_error("file does not exist");

                ArticulationDesc data = MJCFLoader::load(xmlPath.string());
                auto bridgeAsset =
                    ArticulationVisualBridgeAsset::fromData(data);

                const std::string meshBase = "/mesh_assets/mimickit/" + token;
                bridgeAsset.defineMeshAssets(getScene(), meshBase);
                ArticulationVisualBridge skel = bridgeAsset.instantiate(
                    getScene(), "/mimickit/" + token, meshBase);

                glm::vec4 color = assetColor(i);
                std::vector<Scene::Prim*> bodyPrims = skel.bodyPrims();
                for (auto* prim : bodyPrims) {
                    prim->setDisplayColorAlpha(color);
                    getSceneRenderSystem().addRenderable(*prim,
                                                         &commonMaterial);
                }

                _artics.push_back(Articulation::build(
                    physics, data.skeletonTree, data.collisionGeoms,
                    data.joints, data.inertials,
                    ArticulationConfig::freeBase()));
                Articulation& artic = _artics.back();
                physicsBridge.add(artic, skel);

                auto colPrims = physicsBridge.addCollisionVisuals(
                    artic, getScene(), "/collision/mimickit/" + token,
                    showCollision);
                for (auto* prim : colPrims)
                    getSceneRenderSystem().addRenderable(*prim,
                                                         &commonMaterial);

                int col = i % cols;
                int row = i / cols;
                float x = (static_cast<float>(col) - 1.5f) * spacing;
                float y = static_cast<float>(row) * spacing;
                float z = spawnHeight(spec.label);
                float tilt = (i % 2 == 0) ? 0.08f : -0.08f;

                _loaded.push_back({spec, static_cast<int>(_artics.size()) - 1,
                                   artic.numLinks(), artic.numDofs(), x, y, z,
                                   tilt, color, std::move(bodyPrims)});

                std::cout << "[OK] " << spec.label << "  " << xmlPath
                          << "  links=" << artic.numLinks()
                          << " dofs=" << artic.numDofs() << "\n";
            } catch (const std::exception& e) {
                _failed.push_back({spec, e.what()});
                std::cout << "[FAIL] " << spec.label << "  " << xmlPath
                          << "  reason=" << e.what() << "\n";
            }
        }
    }

    glm::vec4 assetColor(int idx) const {
        static const std::array<glm::vec4, 7> colors = {
            glm::vec4(0.30f, 0.65f, 1.00f, 1.0f),
            glm::vec4(1.00f, 0.55f, 0.35f, 1.0f),
            glm::vec4(0.70f, 0.50f, 1.00f, 1.0f),
            glm::vec4(0.35f, 0.85f, 0.55f, 1.0f),
            glm::vec4(1.00f, 0.80f, 0.35f, 1.0f),
            glm::vec4(0.90f, 0.45f, 0.75f, 1.0f),
            glm::vec4(0.60f, 0.85f, 0.95f, 1.0f),
        };
        return colors[static_cast<size_t>(idx) % colors.size()];
    }

    float spawnHeight(const std::string& label) const {
        if (label == "so101")
            return 0.25f;
        if (label == "go2")
            return 0.6f;
        if (label == "smpl")
            return 1.0f;
        return 1.2f;
    }

    void resetAssets() {
        for (const LoadedAsset& item : _loaded) {
            Articulation& artic =
                _artics[static_cast<size_t>(item.articulationIndex)];
            artic.resetRoot(PxTransform(PxVec3(item.x, item.y, item.z),
                                        PxQuat(item.tilt, PxVec3(0.f, 1.f, 0.f))));
            std::vector<float> zeros(static_cast<size_t>(artic.numDofs()), 0.f);
            artic.setDofState(zeros, zeros);
        }
    }

    void setShowCollision(bool enabled) {
        showCollision = enabled;
        physicsBridge.setCollisionVisible(showCollision);
        const float visualAlpha = showCollision ? 0.22f : 1.0f;
        for (const LoadedAsset& item : _loaded) {
            glm::vec4 color = item.color;
            color.a = visualAlpha;
            for (auto* prim : item.bodyPrims) {
                if (prim)
                    prim->setDisplayColorAlpha(color);
            }
        }
    }

    void preUpdate() override {
        bool rDown = glfwGetKey(getWindow(), GLFW_KEY_R) == GLFW_PRESS;
        if (rDown && !rWasDown)
            resetAssets();
        rWasDown = rDown;

        bool cDown = glfwGetKey(getWindow(), GLFW_KEY_C) == GLFW_PRESS;
        if (cDown && !cWasDown) {
            setShowCollision(!showCollision);
        }
        cWasDown = cDown;
    }

    void fixedUpdate(double) override { physics.step(); }

    void preRender() override {
        physicsBridge.sync();
        checkError();
    }

    void render() override {
        ImGui::Begin("MJCF Articulation Gallery");
        ImGui::Text("Loaded: %d / %d", static_cast<int>(_loaded.size()),
                    static_cast<int>(_assets.size()));
        ImGui::Text("Failed: %d  |  %s", static_cast<int>(_failed.size()),
                    isSimulationPaused() ? "PAUSED" : "running");
        ImGui::TextWrapped(
            "Passive free-base articulations: start simulation to run the "
            "ragdoll gravity and collision test.");
        ImGui::Text("Enter: pause/resume    Space: pause/step");
        ImGui::Text("R: reset    C: collision");
        bool collisionVisible = showCollision;
        if (ImGui::Checkbox("Show collision prims", &collisionVisible))
            setShowCollision(collisionVisible);
        ImGui::Separator();

        for (const auto& item : _loaded) {
            ImGui::Text("[OK] %s  links=%d dofs=%d", item.spec.label.c_str(),
                        item.numLinks, item.numDofs);
        }

        if (!_failed.empty()) {
            ImGui::Separator();
            for (const auto& item : _failed) {
                ImGui::TextColored(ImVec4(1.f, 0.35f, 0.25f, 1.f), "[FAIL] %s",
                                   item.spec.label.c_str());
                ImGui::TextWrapped("%s", item.reason.c_str());
            }
        }

        ImGui::End();
    }
};

int main(int argc, char** argv) {
    std::vector<AssetSpec> assets;
    for (int i = 1; i < argc; ++i) {
        fs::path path = fs::absolute(fs::path(argv[i]));
        assets.push_back({path.stem().string(), path.string()});
    }

    if (assets.empty()) {
        assets = {
            {"humanoid", "humanoid/humanoid.xml"},
            {"humanoid_sword_shield", "sword_shield/humanoid_sword_shield.xml"},
            {"smpl", "smpl/smpl.xml"},
            {"g1", "g1/g1.xml"},
            {"g1_mesh", "g1/g1_mesh.xml"},
            {"go2", "go2/go2.xml"},
            {"pi_plus", "hightorque_pi_plus/pi_22dof.xml"},
        };
    }

    MjcfArticulationGalleryApp app(std::move(assets));
    app.initialize(1920, 1080, false, UpAxis::Z);
    app.start();
    return 0;
}
