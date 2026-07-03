#ifndef _INTERACTION_CONTROLLER_HPP_
#define _INTERACTION_CONTROLLER_HPP_

#include <cmath>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "geometry/ray.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"

struct ImDrawList;

namespace KE {

class Camera;

enum class InteractionMode {
    Inspect, // Pick/hover/debug only.
    Edit,    // SceneGraph transform gizmo.
    Force,   // Physics/external-buffer drag force.
    // IK       // IK target / effector controls.
};

class SelectionController {
  public:
    const RayPickResult& lastPick() const { return _lastPick; }
    void setLastPick(const RayPickResult& result) { _lastPick = result; }

    const RayPickResult& selection() const { return _selection; }
    bool hasSelection() const { return _selection.hit; }
    void select(const RayPickResult& result) { _selection = result; }
    void clearSelection() { _selection = RayPickResult{}; }

  private:
    RayPickResult _lastPick;
    RayPickResult _selection;
};

class InteractionController {
  public:
    InteractionMode mode() const { return _mode; }
    void setMode(InteractionMode mode) { _mode = mode; }

    SelectionController& selectionController() { return _selection; }
    const SelectionController& selectionController() const {
        return _selection;
    }

    const RayPickResult& lastPick() const { return _selection.lastPick(); }
    void setLastPick(const RayPickResult& result) {
        _selection.setLastPick(result);
    }

    const RayPickResult& selection() const { return _selection.selection(); }
    bool hasSelection() const { return _selection.hasSelection(); }
    void select(const RayPickResult& result) { _selection.select(result); }
    void clearSelection() { _selection.clearSelection(); }
    bool hasEditableSelection() const {
        return _mode == InteractionMode::Edit && _selection.hasSelection();
    }

    bool handlePick(const RayPickResult& pick,
                    const glm::vec3& forcePlaneNormal) {
        setLastPick(pick);
        if (pick.hit)
            select(pick);
        if (_mode != InteractionMode::Force || !pick.hit)
            return false;
        beginForceDrag(pick, pick.position, forcePlaneNormal);
        return true;
    }

    void handleHoverPick(const RayPickResult& pick) { setLastPick(pick); }

    bool isForceDragActive() const { return _forceDragActive; }
    const RayPickResult& forceDragPick() const { return _forceDragPick; }
    const glm::vec3& forceDragPlanePoint() const {
        return _forceDragPlanePoint;
    }
    const glm::vec3& forceDragPlaneNormal() const {
        return _forceDragPlaneNormal;
    }

    void beginForceDrag(const RayPickResult& pick, const glm::vec3& planePoint,
                        const glm::vec3& planeNormal) {
        _forceDragActive = true;
        _forceDragPick = pick;
        _forceDragPlanePoint = planePoint;
        _forceDragPlaneNormal = planeNormal;
    }

    void endForceDrag() {
        _forceDragActive = false;
        _forceDragPick = RayPickResult{};
    }

    bool cancelActiveInteraction(bool& endedForceDrag) {
        endedForceDrag = false;
        bool consumed = false;
        if (_forceDragActive) {
            endForceDrag();
            endedForceDrag = true;
            consumed = true;
        }
        if (_selection.hasSelection()) {
            clearSelection();
            consumed = true;
        }
        return consumed;
    }

    bool forceDragTarget(const Geometry::Ray& ray, glm::vec3& outTarget) const {
        if (!_forceDragActive)
            return false;
        const float denom = glm::dot(_forceDragPlaneNormal, ray.direction);
        if (std::abs(denom) < 1e-6f)
            return false;
        const float t =
            glm::dot(_forceDragPlanePoint - ray.origin, _forceDragPlaneNormal) /
            denom;
        if (t < 0.0f)
            return false;
        outTarget = ray.getPoint(t);
        return true;
    }

  private:
    InteractionMode _mode = InteractionMode::Inspect;
    SelectionController _selection;
    bool _forceDragActive = false;
    RayPickResult _forceDragPick;
    glm::vec3 _forceDragPlanePoint = glm::vec3(0.0f);
    glm::vec3 _forceDragPlaneNormal = glm::vec3(0.0f, 0.0f, 1.0f);
};

class GizmoController {
  public:
    bool isUsing() const;
    bool manipulateTransform(Camera& camera, glm::mat4& transform,
                             float x, float y, float width,
                             float height, ImDrawList* drawList = nullptr) const;
};

} // namespace KE

#endif
