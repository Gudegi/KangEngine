#include "engine/core/app/interaction_controller.hpp"

#include <imgui.h>
#include <ImGuizmo.h>
#include <glm/gtc/type_ptr.hpp>

#include "engine/scene/types/camera.hpp"

namespace KE {

bool GizmoController::isUsing() const { return ImGuizmo::IsUsing(); }

bool GizmoController::manipulateTransform(Camera& camera, glm::mat4& transform,
                                          float x, float y, float width,
                                          float height,
                                          ImDrawList* drawList) const {
    ImGuizmo::SetOrthographic(false);
    if (drawList)
        ImGuizmo::SetDrawlist(drawList);
    ImGuizmo::SetRect(x, y, width, height);

    const glm::mat4 view = camera.getViewMatrix();
    const glm::mat4 proj = camera.getProjMatrix();
    ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE | ImGuizmo::ROTATE;
    switch (_operation) {
    case GizmoOperation::Translate:
        operation = ImGuizmo::TRANSLATE;
        break;
    case GizmoOperation::Rotate:
        operation = ImGuizmo::ROTATE;
        break;
    case GizmoOperation::Scale:
        operation = ImGuizmo::SCALE;
        break;
    case GizmoOperation::TranslateRotate:
        operation = ImGuizmo::TRANSLATE | ImGuizmo::ROTATE;
        break;
    case GizmoOperation::Universal:
        operation = ImGuizmo::TRANSLATE | ImGuizmo::ROTATE | ImGuizmo::SCALE;
        break;
    }
    const ImGuizmo::MODE mode =
        _space == GizmoSpace::Local ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
    return ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj),
                                operation, mode, glm::value_ptr(transform));
}

} // namespace KE
