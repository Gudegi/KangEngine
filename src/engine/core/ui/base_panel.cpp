#include "base_panel.hpp"
#include "imgui.h"
#include "engine/core/app/app.hpp"
#include <GLFW/glfw3.h>

namespace KE {

BasePanel::BasePanel() {}

BasePanel::~BasePanel() {}

void BasePanel::buildPanel() {
    // ImGui::Begin("Performance");
    ImGui::Text("Performance");
    ImGui::Separator();
    ImGui::Text("FPS: %.1f (%.3f ms/frame)", ImGui::GetIO().Framerate,
                1000.0f / ImGui::GetIO().Framerate);
    // ImGui::Spacing();
    // ImGui::End();
}

MenuBarPanel::MenuBarPanel(App* app) : _app(app) {}

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
                    glfwSetWindowShouldClose(_app->getWindow(), true);
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

ScenePanel::ScenePanel(App* app) : _app(app) {}

ScenePanel::~ScenePanel() {}

void ScenePanel::buildPanel() {
    ImGui::Begin("Scene");
    if (auto* root = _app->getScene()->getRootPrim()) {
        auto drawPrimTree = [&](auto& self, Scene::Prim* prim) -> void {
            const auto& children = prim->getChildren();
            if (children.empty()) {
                ImGui::Text("%s", prim->getName().c_str());
            } else if (ImGui::TreeNode(prim->getName().c_str())) {
                for (auto* child : children)
                    self(self, child);
                ImGui::TreePop();
            }
        };
        for (auto* child : root->getChildren())
            drawPrimTree(drawPrimTree, child);
    }
    ImGui::End();
}

} // namespace KE
