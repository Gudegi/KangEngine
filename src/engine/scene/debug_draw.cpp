#include "engine/scene/debug_draw.hpp"
#include "engine/core/app/app.hpp"
#include "engine/scene/native/prim.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <stdexcept>

namespace KE {
namespace Scene {
namespace {

glm::vec4 pickColor(const std::vector<glm::vec4>& colors, size_t i) {
    if (colors.empty())
        return glm::vec4(1.0f);
    if (colors.size() == 1)
        return colors.front();
    return colors[std::min(i, colors.size() - 1)];
}

void validateLineInputs(const char* functionName,
                        const std::vector<glm::vec3>& starts,
                        const std::vector<glm::vec3>& ends,
                        const std::vector<glm::vec4>& colors) {
    if (starts.size() != ends.size()) {
        throw std::invalid_argument(std::string(functionName) +
                                    " requires starts and ends to have the "
                                    "same length.");
    }
    if (!colors.empty() && colors.size() != 1 &&
        colors.size() != starts.size()) {
        throw std::invalid_argument(std::string(functionName) +
                                    " colors must be empty, length 1, or "
                                    "match the number of elements.");
    }
}

glm::quat rotationFromYTo(const glm::vec3& dir) {
    Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(
        Eigen::Vector3f::UnitY(), Eigen::Vector3f(dir.x, dir.y, dir.z));
    return glm::normalize(glm::quat(q.w(), q.x(), q.y(), q.z()));
}

} // namespace

bool DebugDraw::makeArrowTransform(const glm::vec3& start,
                                   const glm::vec3& end,
                                   glm::mat4& transform) {
    glm::vec3 diff = end - start;
    float len = glm::length(diff);
    if (len < 1e-6f)
        return false;

    glm::vec3 dir = diff / len;
    transform = glm::translate(glm::mat4(1.0f), start) *
                glm::mat4_cast(rotationFromYTo(dir)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, len, 1.0f));
    return true;
}

bool DebugDraw::makeLineTransform(const glm::vec3& start,
                                  const glm::vec3& end,
                                  glm::mat4& transform) {
    glm::vec3 diff = end - start;
    float len = glm::length(diff);
    if (len < 1e-6f)
        return false;

    glm::vec3 dir = diff / len;
    glm::vec3 center = (start + end) * 0.5f;

    transform = glm::translate(glm::mat4(1.0f), center) *
                glm::mat4_cast(rotationFromYTo(dir)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, len, 1.0f));
    return true;
}

std::vector<Prim*> DebugDraw::logLines(SceneBackend* scene,
                                       const std::string& basePath,
                                       const std::vector<glm::vec3>& starts,
                                       const std::vector<glm::vec3>& ends,
                                       const std::vector<glm::vec4>& colors,
                                       float radius, int segments) {
    std::vector<Prim*> result;
    if (!scene)
        return result;

    validateLineInputs("DebugDraw::logLines", starts, ends, colors);

    const size_t lineCount = starts.size();
    result.reserve(lineCount);

    const float safeRadius = std::max(radius, 1e-5f);
    auto meshData = std::make_shared<MeshData>(
        Prim::createCapsuleData(safeRadius, 1.0f, UpAxis::Y, segments));

    for (size_t i = 0; i < lineCount; ++i) {
        glm::vec3 diff = ends[i] - starts[i];
        float len = glm::length(diff);
        if (len < 1e-6f)
            continue;

        glm::vec3 dir = diff / len;
        glm::vec3 center = (starts[i] + ends[i]) * 0.5f;

        auto* prim = scene->definePrim(basePath + "/" + std::to_string(i),
                                       PrimType::Mesh);
        prim->setMeshData(meshData);
        prim->addTranslateOp(center);
        prim->addRotateQuaternionOp(rotationFromYTo(dir));
        prim->addScaleOp(glm::vec3(1.0f, len, 1.0f));
        prim->setDisplayColorAlpha(pickColor(colors, i));
        result.push_back(prim);
    }

    return result;
}

std::vector<Prim*> DebugDraw::logArrows(SceneBackend* scene,
                                        const std::string& basePath,
                                        const std::vector<glm::vec3>& starts,
                                        const std::vector<glm::vec3>& ends,
                                        const std::vector<glm::vec4>& colors,
                                        float radius, int segments) {
    std::vector<Prim*> result;
    if (!scene)
        return result;

    validateLineInputs("DebugDraw::logArrows", starts, ends, colors);

    const size_t arrowCount = starts.size();
    result.reserve(arrowCount);

    const float safeRadius = std::max(radius, 1e-5f);
    auto meshData = std::make_shared<MeshData>(Prim::createArrowData(
        safeRadius, 0.78f, UpAxis::Y, safeRadius * 2.4f, 0.22f, segments));

    for (size_t i = 0; i < arrowCount; ++i) {
        glm::mat4 transform(1.0f);
        if (!DebugDraw::makeArrowTransform(starts[i], ends[i], transform))
            continue;

        auto* prim = scene->definePrim(basePath + "/" + std::to_string(i),
                                       PrimType::Mesh);
        prim->setMeshData(meshData);
        prim->addTranslateOp(starts[i]);
        prim->addRotateQuaternionOp(
            rotationFromYTo(glm::normalize(ends[i] - starts[i])));
        prim->addScaleOp(
            glm::vec3(1.0f, glm::length(ends[i] - starts[i]), 1.0f));
        prim->setDisplayColorAlpha(pickColor(colors, i));
        result.push_back(prim);
    }

    return result;
}

std::vector<Prim*>
DebugDraw::logCoordinateAxes(SceneBackend* scene, const std::string& basePath,
                             glm::vec3 origin, glm::quat orientation,
                             float length, float radius, int segments) {
    const float safeLength = std::max(length, 1e-4f);
    const glm::mat3 basis = glm::mat3_cast(glm::normalize(orientation));

    std::vector<glm::vec3> starts = {origin, origin, origin};
    std::vector<glm::vec3> ends = {
        origin + basis * glm::vec3(safeLength, 0.0f, 0.0f),
        origin + basis * glm::vec3(0.0f, safeLength, 0.0f),
        origin + basis * glm::vec3(0.0f, 0.0f, safeLength),
    };
    std::vector<glm::vec4> colors = {
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
    };

    return logLines(scene, basePath, starts, ends, colors, radius, segments);
}

MeshHandle DebugDraw::logLines(App* app, Backend::Shader* shader,
                               const std::string& path,
                               const std::vector<glm::vec3>& starts,
                               const std::vector<glm::vec3>& ends,
                               const std::vector<glm::vec4>& colors,
                               float radius, int segments) {
    if (!app || !shader || !app->getScene())
        return InvalidHandle;

    validateLineInputs("DebugDraw::logLines", starts, ends, colors);

    const size_t lineCount = starts.size();
    std::vector<glm::mat4> transforms;
    std::vector<glm::vec4> instanceColors;
    transforms.reserve(lineCount);
    instanceColors.reserve(lineCount);

    for (size_t i = 0; i < lineCount; ++i) {
        glm::mat4 transform(1.0f);
        if (!DebugDraw::makeLineTransform(starts[i], ends[i], transform))
            continue;
        transforms.push_back(transform);
        instanceColors.push_back(pickColor(colors, i));
    }

    if (transforms.empty())
        return InvalidHandle;

    const float safeRadius = std::max(radius, 1e-5f);
    auto* prim = app->getScene()->definePrim(path, PrimType::Mesh);
    prim->setMeshData(std::make_shared<MeshData>(
        Prim::createCapsuleData(safeRadius, 1.0f, UpAxis::Y, segments)));

    MeshHandle handle = app->addShape(shader, prim);
    if (handle == InvalidHandle)
        return InvalidHandle;

    app->updateShapeTransforms(handle, transforms, &instanceColors);
    app->setShapeCastsShadow(handle, false);
    return handle;
}

void DebugDraw::updateLines(App* app, MeshHandle handle,
                            const std::vector<glm::vec3>& starts,
                            const std::vector<glm::vec3>& ends,
                            const std::vector<glm::vec4>& colors) {
    if (!app || handle == InvalidHandle)
        return;

    validateLineInputs("DebugDraw::updateLines", starts, ends, colors);

    const size_t lineCount = starts.size();
    std::vector<glm::mat4> transforms;
    std::vector<glm::vec4> instanceColors;
    transforms.reserve(lineCount);
    instanceColors.reserve(lineCount);

    for (size_t i = 0; i < lineCount; ++i) {
        glm::mat4 transform(1.0f);
        if (!DebugDraw::makeLineTransform(starts[i], ends[i], transform))
            continue;
        transforms.push_back(transform);
        instanceColors.push_back(pickColor(colors, i));
    }

    app->updateShapeTransforms(handle, transforms, &instanceColors);
}

MeshHandle DebugDraw::logArrows(App* app, Backend::Shader* shader,
                                const std::string& path,
                                const std::vector<glm::vec3>& starts,
                                const std::vector<glm::vec3>& ends,
                                const std::vector<glm::vec4>& colors,
                                float radius, int segments) {
    if (!app || !shader || !app->getScene())
        return InvalidHandle;

    validateLineInputs("DebugDraw::logArrows", starts, ends, colors);

    const size_t arrowCount = starts.size();
    std::vector<glm::mat4> transforms;
    std::vector<glm::vec4> instanceColors;
    transforms.reserve(arrowCount);
    instanceColors.reserve(arrowCount);

    for (size_t i = 0; i < arrowCount; ++i) {
        glm::mat4 transform(1.0f);
        if (!DebugDraw::makeArrowTransform(starts[i], ends[i], transform))
            continue;
        transforms.push_back(transform);
        instanceColors.push_back(pickColor(colors, i));
    }

    if (transforms.empty())
        return InvalidHandle;

    const float safeRadius = std::max(radius, 1e-5f);
    auto* prim = app->getScene()->definePrim(path, PrimType::Mesh);
    prim->setMeshData(std::make_shared<MeshData>(Prim::createArrowData(
        safeRadius, 0.78f, UpAxis::Y, safeRadius * 2.4f, 0.22f, segments)));

    MeshHandle handle = app->addShape(shader, prim);
    if (handle == InvalidHandle)
        return InvalidHandle;

    app->updateShapeTransforms(handle, transforms, &instanceColors);
    app->setShapeCastsShadow(handle, false);
    return handle;
}

MeshHandle DebugDraw::logCoordinateAxes(App* app, Backend::Shader* shader,
                                        const std::string& path,
                                        glm::vec3 origin, glm::quat orientation,
                                        float length, float radius,
                                        int segments) {
    const float safeLength = std::max(length, 1e-4f);
    const glm::mat3 basis = glm::mat3_cast(glm::normalize(orientation));

    std::vector<glm::vec3> starts = {origin, origin, origin};
    std::vector<glm::vec3> ends = {
        origin + basis * glm::vec3(safeLength, 0.0f, 0.0f),
        origin + basis * glm::vec3(0.0f, safeLength, 0.0f),
        origin + basis * glm::vec3(0.0f, 0.0f, safeLength),
    };
    std::vector<glm::vec4> colors = {
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
    };

    return logLines(app, shader, path, starts, ends, colors, radius, segments);
}

} // namespace Scene
} // namespace KE
