#include "engine/scene/debug_draw.hpp"
#include "engine/core/app/app.hpp"
#include "engine/scene/native/prim.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

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

glm::vec3 vec3At(const float* data, size_t i) {
    const float* p = data + i * 3;
    return glm::vec3(p[0], p[1], p[2]);
}

glm::vec4 vec4At(const float* data, size_t i) {
    const float* p = data + i * 4;
    return glm::vec4(p[0], p[1], p[2], p[3]);
}

glm::vec4 pickColor(const float* colors, size_t colorCount, size_t i) {
    if (!colors || colorCount == 0)
        return glm::vec4(1.0f);
    if (colorCount == 1)
        return vec4At(colors, 0);
    return vec4At(colors, std::min(i, colorCount - 1));
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

void validateRawLineInputs(const char* functionName, const float* starts,
                           const float* ends, const float* colors,
                           size_t count, size_t colorCount) {
    if (count > 0 && (!starts || !ends)) {
        throw std::invalid_argument(std::string(functionName) +
                                    " requires non-null starts and ends when "
                                    "count is non-zero.");
    }
    if (colorCount > 0 && !colors) {
        throw std::invalid_argument(std::string(functionName) +
                                    " requires non-null colors when "
                                    "colorCount is non-zero.");
    }
    if (colorCount != 0 && colorCount != 1 && colorCount != count) {
        throw std::invalid_argument(std::string(functionName) +
                                    " colorCount must be 0, 1, or match "
                                    "count.");
    }
}

glm::quat rotationFromYTo(const glm::vec3& dir) {
    Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(
        Eigen::Vector3f::UnitY(), Eigen::Vector3f(dir.x, dir.y, dir.z));
    return glm::normalize(glm::quat(q.w(), q.x(), q.y(), q.z()));
}

using TransformFn = bool (*)(const glm::vec3&, const glm::vec3&, glm::mat4&);

template <typename StartAt, typename EndAt, typename ColorAt>
std::pair<std::vector<glm::mat4>, std::vector<glm::vec4>>
buildTransforms(size_t count, StartAt startAt, EndAt endAt, ColorAt colorAt,
                TransformFn makeFn) {
    std::vector<glm::mat4> transforms;
    std::vector<glm::vec4> instanceColors;
    transforms.reserve(count);
    instanceColors.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        glm::mat4 t(1.0f);
        if (!makeFn(startAt(i), endAt(i), t))
            continue;
        transforms.push_back(t);
        instanceColors.push_back(colorAt(i));
    }
    return {std::move(transforms), std::move(instanceColors)};
}

std::pair<std::vector<glm::mat4>, std::vector<glm::vec4>>
buildTransforms(const std::vector<glm::vec3>& starts,
                const std::vector<glm::vec3>& ends,
                const std::vector<glm::vec4>& colors, TransformFn makeFn) {
    return buildTransforms(
        starts.size(), [&](size_t i) { return starts[i]; },
        [&](size_t i) { return ends[i]; },
        [&](size_t i) { return pickColor(colors, i); }, makeFn);
}

std::pair<std::vector<glm::mat4>, std::vector<glm::vec4>>
buildTransforms(const float* starts, const float* ends, const float* colors,
                size_t count, size_t colorCount, TransformFn makeFn) {
    return buildTransforms(
        count, [&](size_t i) { return vec3At(starts, i); },
        [&](size_t i) { return vec3At(ends, i); },
        [&](size_t i) { return pickColor(colors, colorCount, i); }, makeFn);
}

} // namespace

bool DebugDraw::makeArrowTransform(const glm::vec3& start, const glm::vec3& end,
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

bool DebugDraw::makeLineTransform(const glm::vec3& start, const glm::vec3& end,
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
        glm::vec3 diff = ends[i] - starts[i];
        float len = glm::length(diff);
        if (len < 1e-6f)
            continue;

        auto* prim = scene->definePrim(basePath + "/" + std::to_string(i),
                                       PrimType::Mesh);
        prim->setMeshData(meshData);
        prim->addTranslateOp(starts[i]);
        prim->addRotateQuaternionOp(rotationFromYTo(diff / len));
        prim->addScaleOp(glm::vec3(1.0f, len, 1.0f));
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

    auto [transforms, instanceColors] =
        buildTransforms(starts, ends, colors, makeLineTransform);
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

MeshHandle DebugDraw::logLines(App* app, Backend::Shader* shader,
                               const std::string& path, const float* starts,
                               const float* ends, const float* colors,
                               size_t count, size_t colorCount, float radius,
                               int segments) {
    if (!app || !shader || !app->getScene())
        return InvalidHandle;

    validateRawLineInputs("DebugDraw::logLines", starts, ends, colors, count,
                          colorCount);

    auto [transforms, instanceColors] = buildTransforms(
        starts, ends, colors, count, colorCount, makeLineTransform);
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
    auto [transforms, instanceColors] =
        buildTransforms(starts, ends, colors, makeLineTransform);
    app->updateShapeTransforms(handle, transforms, &instanceColors);
}

void DebugDraw::updateLines(App* app, MeshHandle handle, const float* starts,
                            const float* ends, const float* colors,
                            size_t count, size_t colorCount) {
    if (!app || handle == InvalidHandle)
        return;
    validateRawLineInputs("DebugDraw::updateLines", starts, ends, colors, count,
                          colorCount);
    auto [transforms, instanceColors] = buildTransforms(
        starts, ends, colors, count, colorCount, makeLineTransform);
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

    auto [transforms, instanceColors] =
        buildTransforms(starts, ends, colors, makeArrowTransform);
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

MeshHandle DebugDraw::logArrows(App* app, Backend::Shader* shader,
                                const std::string& path, const float* starts,
                                const float* ends, const float* colors,
                                size_t count, size_t colorCount, float radius,
                                int segments) {
    if (!app || !shader || !app->getScene())
        return InvalidHandle;

    validateRawLineInputs("DebugDraw::logArrows", starts, ends, colors, count,
                          colorCount);

    auto [transforms, instanceColors] = buildTransforms(
        starts, ends, colors, count, colorCount, makeArrowTransform);
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

void DebugDraw::updateArrows(App* app, MeshHandle handle,
                             const std::vector<glm::vec3>& starts,
                             const std::vector<glm::vec3>& ends,
                             const std::vector<glm::vec4>& colors) {
    if (!app || handle == InvalidHandle)
        return;
    validateLineInputs("DebugDraw::updateArrows", starts, ends, colors);
    auto [transforms, instanceColors] =
        buildTransforms(starts, ends, colors, makeArrowTransform);
    app->updateShapeTransforms(handle, transforms, &instanceColors);
}

void DebugDraw::updateArrows(App* app, MeshHandle handle, const float* starts,
                             const float* ends, const float* colors,
                             size_t count, size_t colorCount) {
    if (!app || handle == InvalidHandle)
        return;
    validateRawLineInputs("DebugDraw::updateArrows", starts, ends, colors,
                          count, colorCount);
    auto [transforms, instanceColors] = buildTransforms(
        starts, ends, colors, count, colorCount, makeArrowTransform);
    app->updateShapeTransforms(handle, transforms, &instanceColors);
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
