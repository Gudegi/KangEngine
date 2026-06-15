#include "kangEngine.hpp"
#include "bridge/skeleton_visual_bridge.hpp"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

using namespace KE;
using namespace KE::Bridge;

namespace {

std::string defaultBvhPath() {
    return KE::getAssetPath(
        "external/SMPL_AMASS_T_HDM_bk_01-01_01_120_poses.bvh");
}

glm::vec4 presetColor(ColorType type, float alpha = 1.0f) {
    const Color& color = ColorLibrary::get(type);
    return glm::vec4(color.r, color.g, color.b, alpha);
}

} // namespace

class BvhCharacterCppApp : public App {
  public:
    std::string bvhPath = defaultBvhPath();
    float importScale = 1.0f;

    std::unique_ptr<Backend::Shader> skeletonShader;
    std::unique_ptr<Backend::Shader> groundShader;
    Animation::SkeletonMotion motion;
    SkeletonVisualBridge skeletonVisualBridge;
    MotionSequencerPanel motionPanel;

    bool animate = true;
    bool showSkeleton = true;
    bool showJoints = true;
    bool spaceWasDown = false;
    bool lWasDown = false;
    bool jWasDown = false;
    float time = 0.0f;

    void setup() override {
        getCamera().setCameraPos(glm::vec3(0.0f, 1.2f, 3.0f));
        getCamera().setTargetPos(glm::vec3(0.0f, 0.9f, 0.0f));
        getRenderer().setLight(
            DirectionalLight{glm::normalize(glm::vec3(0.25f, 0.8f, 0.5f)),
                             glm::vec3(1.0f), 0.9f, glm::vec3(0.18f)});

        const std::string commonVS = KE::getAssetPath("shaders/common.vs");
        const std::string commonFS = KE::getAssetPath("shaders/common.fs");
        const std::string checkerFS =
            KE::getAssetPath("shaders/checkerboard.fs");

        skeletonShader =
            getRenderer().device()->createShaderFromFile(commonVS, commonFS);
        groundShader =
            getRenderer().device()->createShaderFromFile(commonVS, checkerFS);

        for (auto* shader : {skeletonShader.get(), groundShader.get()}) {
            shader->use();
            shader->setUniformBlockBinding("cameraUBO", 0);
            shader->setUniformBlockBinding("lightUBO", 1);
            shader->setUniformBlockBinding("shadowUBO", 2);
        }

        groundShader->use();
        groundShader->setVec4("checkerColor1", presetColor(ColorType::WHITE));
        groundShader->setVec4("checkerColor2",
                              presetColor(ColorType::PASTEL_GREEN));

        MeshPrimDesc ground;
        ground.shader = groundShader.get();
        ground.path = "/ground";
        ground.meshData = Scene::Prim::createPlaneData(20.0f, _upAxis);
        ground.doubleSided = true;
        addMeshPrim(std::move(ground));

        motion = loadBVHMotion(bvhPath, importScale, "/bvh_character");
        const std::string motionName =
            motion.motionName().empty()
                ? std::filesystem::path(bvhPath).filename().string()
                : std::filesystem::path(motion.motionName())
                      .filename()
                      .string();

        motionPanel.setMotion(motionName, motion.numFrames(), motion.fps());
        motionPanel.setPlaying(animate);
        motionPanel.setFrameChangedCallback(
            [this](int frame) { applyFrame(frame); });
        motionPanel.setPlayingChangedCallback(
            [this](bool playing) { animate = playing; });

        SkeletonVisualConfig visualConfig;
        visualConfig.boneRadius = 0.008f;
        visualConfig.jointRadius = 0.025f;
        visualConfig.visible = showSkeleton;
        visualConfig.showJoints = showJoints;
        skeletonVisualBridge = SkeletonVisualBridge::define(
            this, skeletonShader.get(), "/bvh_skeleton", motion, 0.0f, true,
            visualConfig);

        std::cout << "BVH C++ character loaded: " << bvhPath << "\n";
        std::cout << "joints=" << motion.numJoints()
                  << " frames=" << motion.numFrames() << " fps=" << motion.fps()
                  << "\n";
        checkError();
    }

    void updateSkeleton(const Animation::SkeletonState& state) {
        skeletonVisualBridge.setVisible(showSkeleton);
        skeletonVisualBridge.setShowJoints(showJoints);
        skeletonVisualBridge.applyState(state);
    }

    void applyFrame(int frame) {
        frame = std::clamp(frame, 0, motion.numFrames() - 1);
        time = static_cast<float>(frame) / motion.fps();
        updateSkeleton(motion.sample(time, motionPanel.loop()));
        motionPanel.setCurrentTime(time);
        time = motionPanel.currentTime();
    }

    void applyVisibility() {
        updateSkeleton(motion.sample(time, motionPanel.loop()));
    }

    void preRender() override {
        const bool spaceDown =
            glfwGetKey(getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasDown) {
            animate = !animate;
            motionPanel.setPlaying(animate);
        }
        spaceWasDown = spaceDown;

        const bool lDown = glfwGetKey(getWindow(), GLFW_KEY_L) == GLFW_PRESS;
        if (lDown && !lWasDown) {
            showSkeleton = !showSkeleton;
            applyVisibility();
        }
        lWasDown = lDown;

        const bool jDown = glfwGetKey(getWindow(), GLFW_KEY_J) == GLFW_PRESS;
        if (jDown && !jWasDown) {
            showJoints = !showJoints;
            applyVisibility();
        }
        jWasDown = jDown;

        if (animate) {
            time += getDeltaTime() * motionPanel.timeScale();
            motionPanel.setCurrentTime(time);
            time = motionPanel.currentTime();
            updateSkeleton(motion.sample(time, motionPanel.loop()));
        }
        checkError();
    }

    void render() override {
        ImGui::Begin("BVH Character C++");
        ImGui::Text("%s", std::filesystem::path(bvhPath).filename().c_str());
        ImGui::Text("joints=%d frames=%d fps=%.2f", motion.numJoints(),
                    motion.numFrames(), motion.fps());
        ImGui::Text("Space: pause/resume    L: skeleton    J: joints");
        if (ImGui::Checkbox("animate", &animate))
            motionPanel.setPlaying(animate);
        if (ImGui::Checkbox("show skeleton", &showSkeleton))
            applyVisibility();
        if (ImGui::Checkbox("show joints", &showJoints))
            applyVisibility();

        const float duration = std::max(motion.duration(), 1e-6f);
        const float playbackDuration = std::max(
            static_cast<float>(motion.numFrames()) / motion.fps(), 1e-6f);
        const float displayTime =
            std::min(std::fmod(time, playbackDuration), duration);
        ImGui::Text("time %.3f / %.3f", displayTime, duration);
        ImGui::End();

        motionPanel.buildPanel();
    }
};

int main(int argc, char** argv) {
    BvhCharacterCppApp app;
    if (argc > 1)
        app.bvhPath = argv[1];
    if (argc > 2)
        app.importScale = std::stof(argv[2]);
    app.initialize(1920, 1080, false, UpAxis::Y);
    app.start();
    return 0;
}
