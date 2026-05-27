#include "kangEngine.hpp"
#include "engine/graphics/renderer/post_processor.hpp"
#include "geometry/bounds.hpp"

#include <GLFW/glfw3.h>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace KE;

namespace {

struct FrustumLines {
    std::vector<glm::vec3> starts;
    std::vector<glm::vec3> ends;
    std::vector<glm::vec4> colors;
};

struct FrustumPoints {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec4> colors;
};

struct DebugBoundsObject {
    Geometry::AABB localBounds;
    glm::mat4 transform = glm::mat4(1.0f);
    glm::vec4 color = glm::vec4(1.0f);
};

void addLine(FrustumLines& lines, const glm::vec3& a, const glm::vec3& b,
             const glm::vec4& color) {
    lines.starts.push_back(a);
    lines.ends.push_back(b);
    lines.colors.push_back(color);
}

glm::vec4 presetColor(ColorType type) {
    const Color& c = ColorLibrary::get(type);
    return glm::vec4(c.r, c.g, c.b, c.a);
}

glm::mat4 composeTransform(const glm::vec3& position,
                           const glm::quat& orientation,
                           const glm::vec3& scale = glm::vec3(1.0f)) {
    return glm::translate(glm::mat4(1.0f), position) *
           glm::mat4_cast(orientation) * glm::scale(glm::mat4(1.0f), scale);
}

void addAABBLines(FrustumLines& lines, const Geometry::AABB& box,
                  const glm::vec4& color) {
    if (!box.isValid())
        return;

    const glm::vec3 mn = box.min;
    const glm::vec3 mx = box.max;
    const glm::vec3 corners[] = {
        {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mx.x, mx.y, mn.z},
        {mn.x, mx.y, mn.z}, {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z},
        {mx.x, mx.y, mx.z}, {mn.x, mx.y, mx.z},
    };

    const int edges[][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
        {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    for (const auto& edge : edges)
        addLine(lines, corners[edge[0]], corners[edge[1]], color);
}

FrustumLines buildFrustumLines(Camera& camera, float nearPlane, float farPlane,
                               const glm::vec4& nearColor,
                               const glm::vec4& farColor,
                               const glm::vec4& sideColor) {
    FrustumLines lines;
    lines.starts.reserve(12);
    lines.ends.reserve(12);
    lines.colors.reserve(12);

    const WorldFrustumPos f = camera.getFrustumPos(nearPlane, farPlane);

    addLine(lines, f.nearLB, f.nearLT, nearColor);
    addLine(lines, f.nearLT, f.nearRT, nearColor);
    addLine(lines, f.nearRT, f.nearRB, nearColor);
    addLine(lines, f.nearRB, f.nearLB, nearColor);

    addLine(lines, f.farLB, f.farLT, farColor);
    addLine(lines, f.farLT, f.farRT, farColor);
    addLine(lines, f.farRT, f.farRB, farColor);
    addLine(lines, f.farRB, f.farLB, farColor);

    addLine(lines, f.nearLB, f.farLB, sideColor);
    addLine(lines, f.nearLT, f.farLT, sideColor);
    addLine(lines, f.nearRT, f.farRT, sideColor);
    addLine(lines, f.nearRB, f.farRB, sideColor);

    return lines;
}

FrustumLines buildFrustumLines(Camera& camera) {
    return buildFrustumLines(camera, camera.getNearPlane(),
                             camera.getFarPlane(),
                             presetColor(ColorType::PASTEL_CYAN),
                             presetColor(ColorType::PASTEL_ORANGE),
                             presetColor(ColorType::PASTEL_YELLOW));
}

std::vector<float> computeCascadeSplits(Camera& camera, int cascadeCount,
                                        float lambda) {
    std::vector<float> splits(static_cast<size_t>(cascadeCount), 0.0f);
    const float nearPlane = std::max(0.001f, camera.getNearPlane());
    const float farPlane = std::max(nearPlane + 0.001f, camera.getFarPlane());
    const float range = farPlane - nearPlane;
    const float ratio = farPlane / nearPlane;

    for (int i = 1; i <= cascadeCount; ++i) {
        const float p =
            static_cast<float>(i) / static_cast<float>(cascadeCount);
        const float logSplit = nearPlane * std::pow(ratio, p);
        const float uniformSplit = nearPlane + range * p;
        splits[static_cast<size_t>(i - 1)] =
            glm::mix(uniformSplit, logSplit, lambda);
    }
    return splits;
}

FrustumLines buildCascadeFrustumLines(Camera& camera, int cascadeCount,
                                      float lambda) {
    FrustumLines out;
    const std::vector<float> splits =
        computeCascadeSplits(camera, cascadeCount, lambda);
    out.starts.reserve(splits.size() * 12);
    out.ends.reserve(splits.size() * 12);
    out.colors.reserve(splits.size() * 12);

    const glm::vec4 colors[] = {
        presetColor(ColorType::PASTEL_BLUE),
        presetColor(ColorType::PASTEL_GREEN),
        presetColor(ColorType::PASTEL_PEACH),
        presetColor(ColorType::PASTEL_PURPLE),
    };

    float cascadeNear = camera.getNearPlane();
    for (size_t i = 0; i < splits.size(); ++i) {
        const glm::vec4 color = colors[std::min(i, size_t(3))];
        FrustumLines lines =
            buildFrustumLines(camera, cascadeNear, splits[i], color, color,
                              glm::vec4(color.r, color.g, color.b, 0.95f));
        out.starts.insert(out.starts.end(), lines.starts.begin(),
                          lines.starts.end());
        out.ends.insert(out.ends.end(), lines.ends.begin(), lines.ends.end());
        out.colors.insert(out.colors.end(), lines.colors.begin(),
                          lines.colors.end());
        cascadeNear = splits[i];
    }
    return out;
}

FrustumPoints buildFrustumPoints(Camera& camera) {
    const WorldFrustumPos f = camera.getFrustumPos();
    const glm::vec4 nearColor(0.2f, 0.95f, 1.0f, 1.0f);
    const glm::vec4 farColor(1.0f, 0.68f, 0.15f, 1.0f);
    return {
        {f.nearLB, f.nearLT, f.nearRB, f.nearRT, f.farLB, f.farLT, f.farRB,
         f.farRT},
        {nearColor, nearColor, nearColor, nearColor, farColor, farColor,
         farColor, farColor},
    };
}

FrustumLines buildAABBLines(const std::vector<DebugBoundsObject>& objects) {
    FrustumLines lines;
    lines.starts.reserve(objects.size() * 12);
    lines.ends.reserve(objects.size() * 12);
    lines.colors.reserve(objects.size() * 12);

    for (const DebugBoundsObject& object : objects) {
        const Geometry::AABB world =
            Geometry::transformAABB(object.localBounds, object.transform);
        addAABBLines(lines, world, object.color);
    }
    return lines;
}

FrustumLines
buildClassifiedAABBLines(const std::vector<DebugBoundsObject>& objects,
                         const Geometry::Frustum& frustum, int& visibleCount,
                         int& culledCount) {
    FrustumLines lines;
    lines.starts.reserve(objects.size() * 12);
    lines.ends.reserve(objects.size() * 12);
    lines.colors.reserve(objects.size() * 12);

    visibleCount = 0;
    culledCount = 0;
    const glm::vec4 visibleColor(0.25f, 1.0f, 0.35f, 1.0f);
    const glm::vec4 culledColor(1.0f, 0.18f, 0.12f, 1.0f);

    for (const DebugBoundsObject& object : objects) {
        const Geometry::AABB world =
            Geometry::transformAABB(object.localBounds, object.transform);
        const bool visible = Geometry::intersects(frustum, world);
        if (visible)
            ++visibleCount;
        else
            ++culledCount;
        addAABBLines(lines, world, visible ? visibleColor : culledColor);
    }
    return lines;
}

} // namespace

class CameraFrustumDebugApp : public App {
  public:
    std::unique_ptr<Backend::Shader> shader;
    std::unique_ptr<Backend::Shader> groundShader;
    std::unique_ptr<Backend::Framebuffer> subjectPreviewFbo;
    PostProcessor subjectPreviewPost;

    Camera subjectCamera;
    MeshHandle subjectBodyHandle = InvalidHandle;
    std::vector<DebugBoundsObject> debugBoundsObjects;

    bool animateSubject = true;
    bool showSubjectBody = true;
    bool showSceneAABB = true;
    bool showCascadeFrustums = true;
    int aabbClassifyMode =
        1; // 0: object original color, 1: subject, 2: observer
    int cascadeCount = 3;
    float cascadeLambda = 0.55f;
    int subjectVisibleAABB = 0;
    int subjectCulledAABB = 0;
    int observerVisibleAABB = 0;
    int observerCulledAABB = 0;
    float subjectTime = 0.0f;
    float subjectFov = 45.0f;
    float subjectNear = 0.1f;
    float subjectFar = 300.0f;
    float subjectAspect = 16.0f / 9.0f;
    int previewWidth = 480;
    int previewHeight = 270;
    bool previewPostProcess = true;

    void setup() override {
        setCameraMoveSpeed(35.0f);
        getCamera().setCameraPos(glm::vec3(32.0f, 18.0f, 42.0f));
        getCamera().setTargetPos(glm::vec3(0.0f, 1.2f, -70.0f));
        getCamera().setFarPlane(500.0f);

        setLight(
            DirectionalLight{glm::normalize(glm::vec3(0.35f, 0.85f, 0.45f)),
                             glm::vec3(1.0f), 0.75f, glm::vec3(0.14f)});

        const std::string commonVS = KE::getAssetPath("shaders/common.vs");
        const std::string commonFS = KE::getAssetPath("shaders/common.fs");
        const std::string checkerFS =
            KE::getAssetPath("shaders/checkerboard.fs");

        shader = getGraphicsDevice()->createShaderFromFile(commonVS, commonFS);
        groundShader =
            getGraphicsDevice()->createShaderFromFile(commonVS, checkerFS);

        for (Backend::Shader* s : {shader.get(), groundShader.get()}) {
            s->use();
            s->setUniformBlockBinding("cameraUBO", 0);
            s->setUniformBlockBinding("lightUBO", 1);
            s->setUniformBlockBinding("shadowUBO", 2);
        }

        groundShader->use();
        groundShader->setVec4("checkerColor1",
                              glm::vec4(0.9f, 0.9f, 0.9f, 1.0f));
        groundShader->setVec4("checkerColor2",
                              glm::vec4(0.55f, 0.65f, 0.58f, 1.0f));

        MeshPrimDesc ground;
        ground.shader = groundShader.get();
        ground.path = "/debug/ground";
        ground.meshData = Scene::Prim::createPlaneData(360.0f, _upAxis);
        ground.doubleSided = false;
        ground.castsShadow = false;
        addMeshPrim(std::move(ground));

        Scene::DebugDraw::logCoordinateAxes(
            this, shader.get(), "/debug/axis", glm::vec3(0, 0, 0),
            glm::quat(1.0, 0.f, 0.f, 0.f), 1.8f, 0.11f);

        addSceneObjects();
        subjectPreviewFbo = getGraphicsDevice()->createFramebuffer(
            {previewWidth, previewHeight, false, false, 0});
        subjectPreviewPost.init(getGraphicsDevice(), previewWidth,
                                previewHeight);
        initializeSubjectCamera();
        createFrustumDebug();
        checkError();
    }

    void preRender() override {
        if (animateSubject)
            subjectTime += getDeltaTime();

        updateSubjectCamera();
        updateFrustumDebug();
        renderSubjectPreview();
    }

    void render() override {
        ImGui::Begin("Camera Frustum Debug");
        ImGui::Text("Observer: current app camera");
        ImGui::Text("Subject: frustum being drawn");
        ImGui::Separator();
        ImGui::Checkbox("Animate subject camera", &animateSubject);
        ImGui::Checkbox("Show subject body", &showSubjectBody);
        ImGui::Checkbox("Show scene AABB", &showSceneAABB);
        ImGui::Checkbox("Show CSM cascades", &showCascadeFrustums);
        ImGui::SliderInt("Cascade count", &cascadeCount, 1, 4);
        ImGui::SliderFloat("Cascade lambda", &cascadeLambda, 0.0f, 1.0f);
        ImGui::Text("AABB color mode");
        ImGui::RadioButton("Original", &aabbClassifyMode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Subject", &aabbClassifyMode, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Observer", &aabbClassifyMode, 2);
        if (showSceneAABB && aabbClassifyMode == 1) {
            ImGui::Text("Subject visible AABB %d / %d", subjectVisibleAABB,
                        subjectVisibleAABB + subjectCulledAABB);
        }
        if (showSceneAABB && aabbClassifyMode == 2) {
            ImGui::Text("Observer visible AABB %d / %d", observerVisibleAABB,
                        observerVisibleAABB + observerCulledAABB);
        }
        ImGui::SliderFloat("Subject FOV", &subjectFov, 15.0f, 90.0f);
        ImGui::SliderFloat("Subject near", &subjectNear, 0.01f, 2.0f);
        ImGui::SliderFloat("Subject far", &subjectFar, 0.5f, 400.0f);
        ImGui::SliderFloat("Subject aspect", &subjectAspect, 0.5f, 2.5f);
        ImGui::Checkbox("Preview post-process", &previewPostProcess);
        if (subjectPreviewFbo) {
            auto* texture = previewPostProcess
                                ? subjectPreviewPost.getResult()
                                : subjectPreviewFbo->getColorTexture();
            if (texture) {
                ImGui::Separator();
                ImGui::Text("Subject camera view");
                ImGui::Image((ImTextureID)(uintptr_t)texture->getNativeHandle(),
                             ImVec2(static_cast<float>(previewWidth),
                                    static_cast<float>(previewHeight)),
                             ImVec2(0, 1), ImVec2(1, 0));
            }
        }
        ImGui::Text("Move observer with the normal camera controls.");
        ImGui::End();
    }

  private:
    void addBoundsObject(const Scene::MeshData& meshData,
                         const glm::vec3& position,
                         const glm::quat& orientation, const glm::vec4& color) {
        debugBoundsObjects.push_back({meshData.computeLocalAABB(),
                                      composeTransform(position, orientation),
                                      color});
    }

    void addSceneObjects() {
        auto addCubeWithBounds =
            [&](const std::string& path, float size, const glm::vec3& position,
                const glm::quat& orientation, const glm::vec4& color) {
                Scene::MeshData mesh = Scene::Prim::createCubeData(size);
                MeshPrimDesc desc;
                desc.shader = shader.get();
                desc.path = path;
                desc.meshData = mesh;
                desc.position = position;
                desc.orientation = orientation;
                desc.color = color;
                addMeshPrim(std::move(desc));
                addBoundsObject(mesh, position, orientation,
                                glm::vec4(color.r, color.g, color.b, 1.0f));
            };

        auto addSphereWithBounds = [&](const std::string& path, float radius,
                                       const glm::vec3& position,
                                       const glm::quat& orientation,
                                       const glm::vec4& color) {
            Scene::MeshData mesh =
                Scene::Prim::createSphereData(radius, 32, 16);
            MeshPrimDesc desc;
            desc.shader = shader.get();
            desc.path = path;
            desc.meshData = mesh;
            desc.position = position;
            desc.orientation = orientation;
            desc.color = color;
            addMeshPrim(std::move(desc));
            addBoundsObject(mesh, position, orientation,
                            glm::vec4(color.r, color.g, color.b, 1.0f));
        };

        const glm::vec4 warm = presetColor(ColorType::PASTEL_PEACH);
        const glm::vec4 cool = presetColor(ColorType::PASTEL_BLUE);
        const glm::vec4 green = presetColor(ColorType::PASTEL_GREEN);
        const glm::vec4 yellow = presetColor(ColorType::PASTEL_YELLOW);
        const glm::vec4 purple = presetColor(ColorType::PASTEL_PURPLE);

        const std::vector<float> rows = {-8.0f,   -16.0f,  -28.0f,
                                         -44.0f,  -66.0f,  -92.0f,
                                         -124.0f, -162.0f, -205.0f};
        const std::vector<float> lanes = {-14.0f, -7.0f, 0.0f, 7.0f, 14.0f};

        int objectId = 0;
        for (size_t row = 0; row < rows.size(); ++row) {
            for (size_t lane = 0; lane < lanes.size(); ++lane) {
                const float size =
                    0.75f + 0.22f * static_cast<float>((row + lane) % 4);
                const glm::vec3 position(
                    lanes[lane] +
                        std::sin(static_cast<float>(row) * 0.8f) * 0.8f,
                    size * 0.5f, rows[row]);
                const glm::quat orientation = glm::angleAxis(
                    glm::radians(17.0f * static_cast<float>(row + lane)),
                    glm::normalize(glm::vec3(0.25f, 1.0f, 0.15f)));
                const glm::vec4 color = ((row + lane) % 3 == 0)   ? warm
                                        : ((row + lane) % 3 == 1) ? cool
                                                                  : green;
                addCubeWithBounds("/debug/csm_box_" +
                                      std::to_string(objectId++),
                                  size, position, orientation, color);
            }
        }

        for (size_t i = 0; i < rows.size(); ++i) {
            const float z = rows[i] - 5.0f;
            const float radius = 0.45f + 0.08f * static_cast<float>(i % 3);
            addSphereWithBounds("/debug/csm_sphere_left_" + std::to_string(i),
                                radius, glm::vec3(-20.0f, radius, z),
                                glm::quat(1.0f, 0.0f, 0.0f, 0.0f), yellow);
            addSphereWithBounds("/debug/csm_sphere_right_" + std::to_string(i),
                                radius, glm::vec3(20.0f, radius, z - 6.0f),
                                glm::quat(1.0f, 0.0f, 0.0f, 0.0f), purple);
        }

        for (int i = 0; i < 8; ++i) {
            const float z = -18.0f - static_cast<float>(i) * 24.0f;
            const float size = 2.2f + 0.25f * static_cast<float>(i % 2);
            addCubeWithBounds("/debug/csm_tower_left_" + std::to_string(i),
                              size, glm::vec3(-28.0f, size * 0.5f, z),
                              glm::angleAxis(glm::radians(12.0f * i),
                                             glm::vec3(0.f, 1.f, 0.f)),
                              cool);
            addCubeWithBounds("/debug/csm_tower_right_" + std::to_string(i),
                              size, glm::vec3(28.0f, size * 0.5f, z - 10.0f),
                              glm::angleAxis(glm::radians(-9.0f * i),
                                             glm::vec3(0.f, 1.f, 0.f)),
                              warm);
        }
    }

    void initializeSubjectCamera() {
        subjectCamera.init(glm::vec3(0.0f, 2.25f, 6.0f),
                           glm::vec3(0.0f, 1.1f, -55.0f), _upAxis);
        subjectFov = subjectCamera.getFoV();
        subjectNear = subjectCamera.getNearPlane();
        subjectFar = 140.0f;
        updateSubjectCamera();
    }

    void updateSubjectCamera() {
        subjectNear = std::max(0.01f, subjectNear);
        subjectFar = std::max(subjectNear + 0.05f, subjectFar);
        subjectAspect = std::max(0.05f, subjectAspect);

        if (animateSubject) {
            const float angle = subjectTime * 0.18f;
            const glm::vec3 target(5.0f * std::sin(angle * 1.2f), 1.1f,
                                   -55.0f + 8.0f * std::cos(angle * 0.7f));
            const glm::vec3 pos(3.0f * std::sin(angle), 2.25f,
                                6.0f + 2.0f * std::cos(angle));
            subjectCamera.setTargetPos(target);
            subjectCamera.setCameraPos(pos);
        }

        subjectCamera.setFoV(subjectFov);
        subjectCamera.setNearPlane(subjectNear);
        subjectCamera.setFarPlane(subjectFar);
        subjectCamera.updateProjMatrix(
            static_cast<int>(subjectAspect * 1000.0f), 1000);
    }

    void createFrustumDebug() {
        updateFrustumDebug();

        subjectBodyHandle = addSphere(
            "/debug/subject_camera_body", 0.12f, subjectCamera.getCameraPos(),
            glm::vec4(1.0f, 0.15f, 0.15f, 1.0f), shader.get());
        setShapeCastsShadow(subjectBodyHandle, false);
    }

    void updateFrustumDebug() {
        const FrustumLines lines = buildFrustumLines(subjectCamera);
        logDebugLines("/debug/subject_camera_frustum", lines.starts, lines.ends,
                      lines.colors, 2.0f);
        glm::mat4 cameraAxes(1.0f);
        cameraAxes[0] = glm::vec4(subjectCamera.getCameraRightDir(), 0.0f);
        cameraAxes[1] = glm::vec4(subjectCamera.getCameraUpDir(), 0.0f);
        cameraAxes[2] = glm::vec4(subjectCamera.getCameraLookDir(), 0.0f);
        cameraAxes[3] = glm::vec4(subjectCamera.getCameraPos(), 1.0f);
        logDebugAxes("/debug/subject_camera_axes", cameraAxes, 0.6f, 2.0f);
        cascadeCount = std::clamp(cascadeCount, 1, 4);
        cascadeLambda = std::clamp(cascadeLambda, 0.0f, 1.0f);
        const FrustumLines cascadeLines = buildCascadeFrustumLines(
            subjectCamera, cascadeCount, cascadeLambda);
        logDebugLines("/debug/subject_camera_cascades", cascadeLines.starts,
                      cascadeLines.ends, cascadeLines.colors, 2.5f,
                      !showCascadeFrustums);
        const FrustumPoints points = buildFrustumPoints(subjectCamera);
        logDebugPoints("/debug/subject_camera_frustum_corners",
                       points.positions, points.colors, 9.0f);

        int visibleAABB = 0;
        int culledAABB = 0;
        const Geometry::Frustum subjectFrustum =
            Geometry::Frustum::fromViewProjection(
                subjectCamera.getProjMatrix() * subjectCamera.getViewMatrix());
        const Geometry::Frustum observerFrustum =
            Geometry::Frustum::fromViewProjection(getProjectionMatrix() *
                                                  getViewMatrix());

        FrustumLines aabbLines;
        if (aabbClassifyMode == 1) {
            aabbLines = buildClassifiedAABBLines(
                debugBoundsObjects, subjectFrustum, visibleAABB, culledAABB);
            subjectVisibleAABB = visibleAABB;
            subjectCulledAABB = culledAABB;
        } else if (aabbClassifyMode == 2) {
            aabbLines = buildClassifiedAABBLines(
                debugBoundsObjects, observerFrustum, visibleAABB, culledAABB);
            observerVisibleAABB = visibleAABB;
            observerCulledAABB = culledAABB;
        } else {
            aabbLines = buildAABBLines(debugBoundsObjects);
        }
        logDebugLines("/debug/scene_aabb", aabbLines.starts, aabbLines.ends,
                      aabbLines.colors, 1.5f, !showSceneAABB);

        if (subjectBodyHandle != InvalidHandle) {
            std::vector<glm::mat4> transforms(
                1,
                glm::translate(glm::mat4(1.0f), subjectCamera.getCameraPos()));
            const float alpha = showSubjectBody ? 1.0f : 0.0f;
            std::vector<glm::vec4> colors{glm::vec4(1.0f, 0.15f, 0.15f, alpha)};
            updateShapeTransforms(subjectBodyHandle, transforms, &colors);
        }
    }

    void renderSubjectPreview() {
        if (!subjectPreviewFbo)
            return;
        renderSceneToFramebuffer(subjectCamera, subjectPreviewFbo.get(),
                                 previewWidth, previewHeight);
        subjectPreviewPost.process(subjectPreviewFbo->getColorTexture(),
                                   _gamma);
    }
};

int main() {
    CameraFrustumDebugApp app;
    app.initialize(1920, 1080, false, UpAxis::Y);
    app.start();
    return 0;
}
