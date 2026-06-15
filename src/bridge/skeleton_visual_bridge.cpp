#include "skeleton_visual_bridge.hpp"
#include "animation/skeleton_math.hpp"
#include "engine/core/app/app.hpp"
#include "engine/scene/debug_draw.hpp"
#include "engine/scene/native/prim.hpp"

#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace KE {
namespace Bridge {

static glm::vec4 withVisibility(glm::vec4 color, bool visible) {
    color.a *= visible ? 1.0f : 0.0f;
    return color;
}

static std::vector<glm::vec4> repeatedColor(size_t count, glm::vec4 color) {
    return std::vector<glm::vec4>(count, color);
}

static std::vector<glm::mat4>
jointTransforms(const std::vector<glm::vec3>& positions, float radius) {
    std::vector<glm::mat4> transforms;
    transforms.reserve(positions.size());
    const glm::vec3 scale(std::max(radius, 1e-5f));
    for (const glm::vec3& p : positions) {
        transforms.push_back(glm::translate(glm::mat4(1.0f), p) *
                             glm::scale(glm::mat4(1.0f), scale));
    }
    return transforms;
}

SkeletonVisualBridge SkeletonVisualBridge::define(
    App* app, Backend::Shader* shader, const std::string& basePath,
    const Animation::SkeletonState& state, const SkeletonVisualConfig& config) {
    SkeletonVisualBridge bridge;
    bridge._app = app;
    bridge._shader = shader;
    bridge._basePath = basePath;
    bridge._config = config;
    bridge.applyState(state);
    return bridge;
}

SkeletonVisualBridge SkeletonVisualBridge::define(
    App* app, Backend::Shader* shader, const std::string& basePath,
    const Animation::SkeletonMotion& motion, float time, bool loop,
    const SkeletonVisualConfig& config) {
    return define(app, shader, basePath, motion.sample(time, loop), config);
}

void SkeletonVisualBridge::applyState(const Animation::SkeletonState& state) {
    _lastState = state;
    if (!_app || !_shader)
        return;

    const Animation::SkeletonBoneLines bones = Animation::boneLines(state);
    const std::vector<glm::vec4> boneColors =
        repeatedColor(bones.starts.size(),
                      withVisibility(_config.boneColor, _config.visible));

    if (_boneHandle == InvalidHandle) {
        _boneHandle = Scene::DebugDraw::logLines(
            _app, _shader, _basePath + "/bones", bones.starts, bones.ends,
            boneColors, _config.boneRadius, _config.segments);
        _app->setRenderableCastsShadow(_boneHandle, false);
    } else {
        Scene::DebugDraw::updateLines(_app, _boneHandle, bones.starts,
                                      bones.ends, boneColors);
    }

    const std::vector<glm::vec3> joints = Animation::jointPositions(state);
    if (_config.showJoints || _jointHandle != InvalidHandle) {
        const std::vector<glm::vec4> jointColors = repeatedColor(
            joints.size(),
            withVisibility(_config.jointColor,
                           _config.visible && _config.showJoints));

        if (_jointHandle == InvalidHandle) {
            App::MeshPrimDesc desc;
            desc.shader = _shader;
            desc.path = _basePath + "/joints";
            desc.meshData = Scene::Prim::createSphereData(1.0f, 16, 8);
            desc.castsShadow = false;
            _jointHandle = _app->addMeshPrim(std::move(desc)).handle;
            _app->setRenderableCastsShadow(_jointHandle, false);
        }

        if (_jointHandle != InvalidHandle) {
            _app->updateRenderableTransforms(
                _jointHandle, jointTransforms(joints, _config.jointRadius),
                &jointColors);
        }
    }
}

void SkeletonVisualBridge::applyMotion(const Animation::SkeletonMotion& motion,
                                       float time, bool loop) {
    applyState(motion.sample(time, loop));
}

void SkeletonVisualBridge::setVisible(bool visible) {
    if (_config.visible == visible)
        return;
    _config.visible = visible;
    if (_lastState)
        applyState(*_lastState);
}

void SkeletonVisualBridge::setShowJoints(bool showJoints) {
    if (_config.showJoints == showJoints)
        return;
    _config.showJoints = showJoints;
    if (_lastState)
        applyState(*_lastState);
}

} // namespace Bridge
} // namespace KE
