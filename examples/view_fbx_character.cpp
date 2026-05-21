#include "kangEngine.hpp"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace KE;
using namespace KE::Bridge;

namespace {

std::string defaultFbxPath() {
    return KE::getAssetPath("external/Capoeira.fbx");
}

std::vector<glm::vec4> skeletonColors(const std::vector<std::string>& names,
                                      const std::vector<int>& parentIndices,
                                      bool visible) {
    std::vector<glm::vec4> colors;
    colors.reserve(parentIndices.size());
    const float alpha = visible ? 1.0f : 0.0f;
    for (int i = 0; i < static_cast<int>(parentIndices.size()); ++i) {
        if (parentIndices[static_cast<size_t>(i)] < 0)
            continue;
        std::string name = names[static_cast<size_t>(i)];
        for (char& ch : name)
            ch =
                static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (name.find("left") != std::string::npos)
            colors.push_back(glm::vec4(0.2f, 0.45f, 1.0f, alpha));
        else if (name.find("right") != std::string::npos)
            colors.push_back(glm::vec4(1.0f, 0.25f, 0.18f, alpha));
        else
            colors.push_back(glm::vec4(0.98f, 0.98f, 0.98f, alpha));
    }
    return colors;
}

} // namespace

class FbxCharacterCppApp : public App {
  public:
    std::string fbxPath = defaultFbxPath();
    int clipIndex = 0;
    float fps = 30.0f;
    float importScale = 0.01f;

    std::unique_ptr<Backend::Shader> skinnedShader;
    std::unique_ptr<Backend::Shader> lineShader;
    std::unique_ptr<Backend::Shader> groundShader;
    SkinnedCharacterBridge character;

    MeshHandle skeletonHandle = InvalidHandle;
    std::vector<glm::vec3> lineStarts;
    std::vector<glm::vec3> lineEnds;
    std::vector<glm::vec4> lineColors;

    bool animate = true;
    bool showMesh = true;
    bool showSkeleton = true;
    bool castShadows = true;
    bool spaceWasDown = false;
    bool mWasDown = false;
    bool lWasDown = false;
    bool hWasDown = false;
    float playbackSpeed = 1.0f;
    float time = 0.0f;

    void setup() override {
        getCamera().setCameraPos(glm::vec3(0.0f, 1.45f, 3.2f));
        getCamera().setTargetPos(glm::vec3(0.0f, 0.85f, 0.0f));
        // DirectionalLight::direction is the Phong L vector pointing toward the
        // light source, so Y-up scenes should use a positive Y component.
        setLight(
            DirectionalLight{glm::normalize(glm::vec3(0.25f, 0.75f, 0.55f)),
                             glm::vec3(1.0f), 0.85f, glm::vec3(0.12f)});

        auto commonVSPath = KE::getAssetPath("shaders/common.vs");
        auto skinnedVSPath = KE::getAssetPath("shaders/skinned_mesh.vs");
        auto commonFSPath = KE::getAssetPath("shaders/common.fs");
        auto commonTexFSPath = KE::getAssetPath("shaders/commonTex.fs");
        auto checkerFSPath = KE::getAssetPath("shaders/checkerboard.fs");

        skinnedShader = getGraphicsDevice()->createShaderFromFile(
            skinnedVSPath, commonTexFSPath);
        lineShader = getGraphicsDevice()->createShaderFromFile(commonVSPath,
                                                               commonFSPath);
        groundShader = getGraphicsDevice()->createShaderFromFile(commonVSPath,
                                                                 checkerFSPath);

        for (auto* shader :
             {skinnedShader.get(), lineShader.get(), groundShader.get()}) {
            shader->use();
            shader->setUniformBlockBinding("cameraUBO", 0);
            shader->setUniformBlockBinding("lightUBO", 1);
            shader->setUniformBlockBinding("shadowUBO", 2);
        }

        groundShader->use();
        groundShader->setVec4("checkerColor1",
                              glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
        groundShader->setVec4("checkerColor2",
                              glm::vec4(0.77f, 0.93f, 0.78f, 1.0f));

        MeshPrimDesc ground;
        ground.shader = groundShader.get();
        ground.path = "/ground";
        ground.meshData = Scene::Prim::createPlaneData(20.0f, _upAxis);
        addMeshPrim(std::move(ground));

        character = SkinnedCharacterBridge::fromFBX(
            this, skinnedShader.get(), fbxPath, "/fbx_character", clipIndex,
            fps, importScale);
        character.setCastsShadow(castShadows);
        updateSkeletonLines(character.motion().sample(0.0f, true));

        std::cout << "FBX C++ character loaded: " << fbxPath << "\n";
        std::cout << "meshes=" << character.meshes().size()
                  << " joints=" << character.motion().numJoints()
                  << " frames=" << character.motion().numFrames() << "\n";
        checkError();
    }

    void updateSkeletonLines(const Animation::SkeletonState& state) {
        const auto positions = state.computeGlobalPositions();
        const auto parents = character.motion().skeletonTree().parentIndices();
        const auto names = character.motion().skeletonTree().nodeNames();

        lineStarts.clear();
        lineEnds.clear();
        lineStarts.reserve(parents.size());
        lineEnds.reserve(parents.size());
        for (int i = 0; i < static_cast<int>(parents.size()); ++i) {
            const int parent = parents[static_cast<size_t>(i)];
            if (parent < 0)
                continue;
            const Eigen::Vector3f& a = positions[static_cast<size_t>(parent)];
            const Eigen::Vector3f& b = positions[static_cast<size_t>(i)];
            lineStarts.emplace_back(a.x(), a.y(), a.z());
            lineEnds.emplace_back(b.x(), b.y(), b.z());
        }

        lineColors = skeletonColors(names, parents, showSkeleton);
        if (skeletonHandle == InvalidHandle) {
            skeletonHandle = Scene::DebugDraw::logLines(
                this, lineShader.get(), "/debug/fbx_skeleton", lineStarts,
                lineEnds, lineColors, 0.008f, 8);
        } else {
            Scene::DebugDraw::updateLines(this, skeletonHandle, lineStarts,
                                          lineEnds, lineColors);
        }
    }

    void applyVisibility() {
        character.setVisible(showMesh);
        updateSkeletonLines(character.motion().sample(time, true));
    }

    void preRender() override {
        const bool spaceDown =
            glfwGetKey(getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasDown)
            animate = !animate;
        spaceWasDown = spaceDown;

        const bool mDown = glfwGetKey(getWindow(), GLFW_KEY_M) == GLFW_PRESS;
        if (mDown && !mWasDown) {
            showMesh = !showMesh;
            applyVisibility();
        }
        mWasDown = mDown;

        const bool lDown = glfwGetKey(getWindow(), GLFW_KEY_L) == GLFW_PRESS;
        if (lDown && !lWasDown) {
            showSkeleton = !showSkeleton;
            applyVisibility();
        }
        lWasDown = lDown;

        const bool hDown = glfwGetKey(getWindow(), GLFW_KEY_H) == GLFW_PRESS;
        if (hDown && !hWasDown) {
            castShadows = !castShadows;
            character.setCastsShadow(castShadows);
        }
        hWasDown = hDown;

        if (animate) {
            time += getDeltaTime() * playbackSpeed;
            const Animation::SkeletonState state = character.applyTime(time);
            updateSkeletonLines(state);
        }
        checkError();
    }

    void render() override {
        ImGui::Begin("FBX Character C++");
        ImGui::Text("%s", std::filesystem::path(fbxPath).filename().c_str());
        ImGui::Text("meshes=%d joints=%d frames=%d",
                    static_cast<int>(character.meshes().size()),
                    character.motion().numJoints(),
                    character.motion().numFrames());
        ImGui::Text(
            "Space: pause/resume    M: mesh    L: skeleton    H: shadow");
        if (ImGui::Checkbox("show mesh", &showMesh))
            applyVisibility();
        if (ImGui::Checkbox("show skeleton", &showSkeleton))
            applyVisibility();
        if (ImGui::Checkbox("cast shadows", &castShadows))
            character.setCastsShadow(castShadows);
        ImGui::Checkbox("animate", &animate);
        ImGui::SliderFloat("playback speed", &playbackSpeed, 0.0f, 4.0f);
        const float duration = std::max(character.motion().duration(), 1e-6f);
        ImGui::Text("time %.3f / %.3f", std::fmod(time, duration), duration);
        ImGui::End();
    }
};

int main(int argc, char** argv) {
    FbxCharacterCppApp app;
    if (argc > 1)
        app.fbxPath = argv[1];
    app.initialize(1920, 1080, false, UpAxis::Y);
    app.start();
    return 0;
}
