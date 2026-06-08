#include "base_panel.hpp"
#include "imgui.h"
#include "engine/core/app/app.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include <cstdio>
#include <cstdint>

namespace KE {

PerformancePanel::PerformancePanel() : Panel("Performance") {}

PerformancePanel::~PerformancePanel() {}

void PerformancePanel::buildPanel() {
    ImGui::Begin(name().c_str());
    ImGui::Text("Performance");
    ImGui::Separator();
    ImGui::Text("FPS: %.1f (%.3f ms/frame)", ImGui::GetIO().Framerate,
                1000.0f / ImGui::GetIO().Framerate);
    ImGui::End();
}

RendererDebugPanel::RendererDebugPanel(App* app)
    : Panel("Renderer Debug"), _app(app) {}

RendererDebugPanel::~RendererDebugPanel() {}

void RendererDebugPanel::buildPanel() {
    if (!_app)
        return;

    ImGui::Begin(name().c_str());
    ImGui::Checkbox("Wireframe", &_app->_renderWireframe);
    const char* interactionLabels[] = {"Inspect", "Edit", "Force"};
    int interactionMode = static_cast<int>(_app->getInteractionMode());
    if (ImGui::Combo("Interaction Mode", &interactionMode, interactionLabels,
                     3)) {
        _app->setInteractionMode(static_cast<InteractionMode>(interactionMode));
    }
    ImGui::SliderFloat("GammaCorrection", &_app->_gamma, 0.f, 5.f);
    if (SelectionOutlineProcessor* outline =
            _app->getSelectionOutlineProcessor()) {
        SelectionOutlineConfig& config = outline->config();
        ImGui::Checkbox("Selection Outline", &config.enabled);
        float outlineColor[4] = {config.color.r, config.color.g, config.color.b,
                                 config.color.a};
        if (ImGui::ColorEdit4("Selection Outline Color", outlineColor)) {
            config.color = glm::vec4(outlineColor[0], outlineColor[1],
                                     outlineColor[2], outlineColor[3]);
        }
        ImGui::SliderFloat("Selection Outline Radius", &config.radius, 1.0f,
                           8.0f, "%.1f px");
    }
    ImGui::DragFloat("Camera Move Speed", &_app->_cameraMoveSpeed, 0.2f, 0.0f,
                     500.0f, "%.2f");

    Rasterizer* rasterizer = _app->getRasterizer();
    if (!rasterizer) {
        ImGui::End();
        return;
    }

    DirectionalLight light = _app->getLight();
    glm::vec3 direction = light.direction;
    float color[3] = {light.color.r, light.color.g, light.color.b};
    glm::vec3 ambient = light.ambient;

    if (ImGui::DragFloat3("Sun Direction (toward light)", &direction.x,
                          0.02f)) {
        _app->setLightDirection(direction);
    }
    if (ImGui::ColorEdit3("Light Color", color)) {
        _app->setLightColor(glm::vec3(color[0], color[1], color[2]));
    }
    if (ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 2.0f)) {
        _app->setLightIntensity(light.intensity);
    }
    if (ImGui::ColorEdit3("Ambient", &ambient.x)) {
        _app->setLightAmbient(ambient);
    }

    float distance = rasterizer->getShadowDistance();
    if (ImGui::SliderFloat("Shadow Distance (Set 0 to disable shadow)",
                           &distance, 0.0f, 300.0f)) {
        rasterizer->setShadowDistance(distance);
    }
    int pcfSamples = rasterizer->getShadowPcfSamples();
    if (ImGui::SliderInt("Shadow PCF Samples", &pcfSamples, 1, 16)) {
        rasterizer->setShadowPcfSamples(pcfSamples);
    }
    bool useCsm = rasterizer->getUseCsm();
    if (ImGui::Checkbox("Use CSM", &useCsm)) {
        rasterizer->setUseCsm(useCsm);
    }
    if (useCsm) {
        int cascadeCount = rasterizer->getCascadeCount();
        if (ImGui::SliderInt("CSM Cascade Count", &cascadeCount, 1,
                             Rasterizer::MaxShadowCascades)) {
            rasterizer->setCascadeCount(cascadeCount);
        }
        float cascadeLambda = rasterizer->getCascadeLambda();
        if (ImGui::SliderFloat("CSM Cascade Lambda", &cascadeLambda, 0.0f,
                               1.0f)) {
            rasterizer->setCascadeLambda(cascadeLambda);
        }
        bool useTightShadowFit = rasterizer->getUseTightShadowFit();
        if (ImGui::Checkbox("Tight Shadow Fit", &useTightShadowFit)) {
            rasterizer->setUseTightShadowFit(useTightShadowFit);
        }
        bool debugCascadeTint = rasterizer->getDebugCsmCascadeTint();
        if (ImGui::Checkbox("Debug CSM Cascade Tint", &debugCascadeTint)) {
            rasterizer->setDebugCsmCascadeTint(debugCascadeTint);
        }
    }

    bool frustumCulling = rasterizer->isFrustumCullingEnabled();
    if (ImGui::Checkbox("Frustum Culling", &frustumCulling)) {
        rasterizer->setFrustumCullingEnabled(frustumCulling);
    }
    bool debugRenderAABB = rasterizer->getDebugRenderAABB();
    if (ImGui::Checkbox("Show Render AABB", &debugRenderAABB)) {
        rasterizer->setDebugRenderAABB(debugRenderAABB);
    }
    if (frustumCulling) {
        // Batch = one instancer/draw group. Instance = one transform inside
        // that batch, culled by its world AABB.
        ImGui::Text("Culled Batches %d / %d",
                    rasterizer->getCullingCulledBatches(),
                    rasterizer->getCullingTotalBatches());
        ImGui::Text("Culled Instances %d / %d",
                    rasterizer->getCullingCulledInstances(),
                    rasterizer->getCullingTotalInstances());
    }

    if (useCsm) {
        const int cascadeCount = rasterizer->getCascadeCount();
        ImGui::Text("CSM Shadow Maps");
        if (ImGui::BeginTable("CSMShadowMapPreview", 2)) {
            for (int i = 0; i < cascadeCount; ++i) {
                auto* cascadeFbo = rasterizer->getCascadeShadowFbo(i);
                if (!cascadeFbo)
                    continue;
                auto* depthTex = cascadeFbo->getDepthTexture();
                if (!depthTex)
                    continue;
                ImGui::TableNextColumn();
                ImGui::Text("Cascade %d %dx%d", i, depthTex->getWidth(),
                            depthTex->getHeight());
                ImGui::Image(
                    (ImTextureID)(uintptr_t)depthTex->getNativeHandle(),
                    ImVec2(128, 128), ImVec2(0, 1), ImVec2(1, 0));
            }
            ImGui::EndTable();
        }
    } else {
        auto* shadowFbo = rasterizer->getShadowFbo();
        if (shadowFbo) {
            auto* depthTex = shadowFbo->getDepthTexture();
            if (depthTex) {
                ImGui::Text("Shadow Map %dx%d", depthTex->getWidth(),
                            depthTex->getHeight());
                ImGui::Image(
                    (ImTextureID)(uintptr_t)depthTex->getNativeHandle(),
                    ImVec2(128, 128), ImVec2(0, 1), ImVec2(1, 0));
            }
        }
    }
    ImGui::End();
}

MenuBarPanel::MenuBarPanel(App* app) : Panel("Menu Bar"), _app(app) {}

MenuBarPanel::~MenuBarPanel() {}

void MenuBarPanel::buildPanel() {
    if (ImGui::BeginMainMenuBar()) {
        // TODO: implement this
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Scene")) {
            }
            if (ImGui::MenuItem("Open...")) {
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Save")) {
            }
            if (ImGui::MenuItem("Exit")) {
                if (_app)
                    _app->requestClose();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Settings")) {
            ImGui::EndMenu();
        }
        /*
        if (ImGui::Button("Play")) {}
        ImGui::SameLine();
        if (ImGui::Button("Pause")) {}
        */

        char fpsText[32];
        std::snprintf(fpsText, sizeof(fpsText), "FPS: %d",
                      int(ImGui::GetIO().Framerate));
        float textWidth = ImGui::CalcTextSize(fpsText).x;
        float rightPadding = ImGui::GetStyle().FramePadding.x;
        float cursorX = ImGui::GetWindowWidth() - textWidth - rightPadding;
        if (cursorX > ImGui::GetCursorPosX()) {
            ImGui::SetCursorPosX(cursorX);
        }
        ImGui::TextDisabled("%s", fpsText);
        ImGui::Spacing();
        // if (ImGui::SmallButton("somthing")) {}

        ImGui::EndMainMenuBar();
    }
}

ScenePanel::ScenePanel(App* app) : Panel("Scene"), _app(app) {}

ScenePanel::~ScenePanel() {}

void ScenePanel::buildPanel() {
    ImGui::Begin(name().c_str());
    if (auto* root = _app->getScene()->getRootPrim()) {
        auto drawPrimTree = [&](auto& self, Scene::Prim* prim) -> void {
            ImGui::PushID(prim);
            // bool active = prim->isActive();
            // if (ImGui::Checkbox("##active", &active))
            //     prim->setActive(active);
            // ImGui::SameLine();
            bool visible = prim->isVisible();
            if (ImGui::Checkbox("##Visible", &visible))
                prim->setVisible(visible);
            ImGui::SameLine();

            const bool activeInHierarchy = prim->isActiveInHierarchy();
            const bool visibleInHierarchy = prim->isVisibleInHierarchy();
            if (!activeInHierarchy || !visibleInHierarchy)
                ImGui::PushStyleColor(
                    ImGuiCol_Text,
                    ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));

            const auto& children = prim->getChildren();
            if (children.empty()) {
                ImGui::Text("%s", prim->getName().c_str());
            } else if (ImGui::TreeNode(prim->getName().c_str())) {
                if (!activeInHierarchy || !visibleInHierarchy)
                    ImGui::PopStyleColor();
                for (auto* child : children)
                    self(self, child);
                ImGui::TreePop();
                ImGui::PopID();
                return;
            }

            if (!activeInHierarchy || !visibleInHierarchy)
                ImGui::PopStyleColor();
            ImGui::PopID();
        };
        // ImGui::TextDisabled("A");
        // ImGui::SameLine();
        ImGui::TextDisabled("Visible");
        ImGui::Separator();
        for (auto* child : root->getChildren())
            drawPrimTree(drawPrimTree, child);
    }
    ImGui::End();
}

} // namespace KE
