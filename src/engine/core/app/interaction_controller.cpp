#include "engine/core/app/interaction_controller.hpp"

#include <imgui.h>
#include <ImGuizmo.h>
#include <glm/gtc/type_ptr.hpp>

#include "engine/graphics/camera/camera.hpp"

namespace KE {

bool GizmoController::manipulateTransform(Camera& camera,
                                          glm::mat4& transform) const {
    ImGuizmo::SetOrthographic(false);
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGuizmo::SetDrawlist(ImGui::GetForegroundDrawList(viewport));
    ImGuizmo::SetRect(viewport->Pos.x, viewport->Pos.y, viewport->Size.x,
                      viewport->Size.y);

    const glm::mat4 view = camera.getViewMatrix();
    const glm::mat4 proj = camera.getProjMatrix();
    ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE | ImGuizmo::ROTATE;
    return ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj),
                                operation, ImGuizmo::WORLD,
                                glm::value_ptr(transform));
}

} // namespace KE
