#include "base_panel.hpp"
#include "imgui.h"

/*
BasePanel::BasePanel() {}

BasePanel::~BasePanel() {}
*/
void BasePanel::buildPanel()
{
    ImGui::Text("Performance");
    ImGui::Separator();
    ImGui::Text("FPS: %.1f (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
    //ImGui::Spacing();
}