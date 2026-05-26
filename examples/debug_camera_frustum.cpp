#include "kangEngine.hpp"
#include "engine/graphics/renderer/post_processor.hpp"
#include "geometry/bounds.hpp"

#include <GLFW/glfw3.h>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>
#include <imgui.h>

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

FrustumLines buildFrustumLines(Camera& camera) {
    FrustumLines lines;
    lines.starts.reserve(12);
    lines.ends.reserve(12);
    lines.colors.reserve(12);

    const WorldFrustumPos f = camera.getFrustumPos();
    const glm::vec4 nearColor(0.2f, 0.95f, 1.0f, 1.0f);
    const glm::vec4 farColor(1.0f, 0.68f, 0.15f, 1.0f);
    const glm::vec4 sideColor(0.95f, 0.95f, 0.2f, 1.0f);

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
    int aabbClassifyMode =
        1; // 0: object original color, 1: subject, 2: observer
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
        setCameraMoveSpeed(8.0f);
        getCamera().setCameraPos(glm::vec3(7.0f, 4.5f, 8.0f));
        getCamera().setTargetPos(glm::vec3(0.0f, 1.0f, 0.0f));
        getCamera().setFarPlane(200.0f);

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
        ground.meshData = Scene::Prim::createPlaneData(24.0f, _upAxis);
        ground.doubleSided = false;
        ground.castsShadow = false;
        addMeshPrim(std::move(ground));

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

        addCubeWithBounds(
            "/debug/box_left", 0.65f, glm::vec3(-2.2f, 0.33f, -1.2f),
            glm::angleAxis(glm::radians(30.0f), glm::vec3(1.f, 0.f, 0.f)),
            glm::vec4(0.95f, 0.34f, 0.25f, 1.0f));
        addCubeWithBounds(
            "/debug/box_right", 0.55f, glm::vec3(2.0f, 0.28f, 1.25f),
            glm::angleAxis(glm::radians(45.0f),
                           glm::normalize(glm::vec3(1.f, 1.f, 0.f))),
            glm::vec4(0.25f, 0.55f, 0.95f, 1.0f));
        addSphereWithBounds(
            "/debug/sphere_center", 0.45f, glm::vec3(0.0f, 0.45f, 0.0f),
            glm::angleAxis(glm::radians(15.0f), glm::vec3(1.f, 0.f, 0.f)),
            glm::vec4(0.95f, 0.82f, 0.25f, 1.0f));
        addSphereWithBounds("/debug/sphere_far", 0.35f,
                            glm::vec3(0.9f, 0.35f, -3.5f),
                            glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                            glm::vec4(0.45f, 0.95f, 0.55f, 1.0f));
    }

    void initializeSubjectCamera() {
        subjectCamera.init(glm::vec3(-3.0f, 1.35f, 3.2f),
                           glm::vec3(0.3f, 0.95f, -0.3f), _upAxis);
        subjectFov = subjectCamera.getFoV();
        subjectNear = subjectCamera.getNearPlane();
        subjectFar = subjectCamera.getFarPlane();
        updateSubjectCamera();
    }

    void updateSubjectCamera() {
        subjectNear = std::max(0.01f, subjectNear);
        subjectFar = std::max(subjectNear + 0.05f, subjectFar);
        subjectAspect = std::max(0.05f, subjectAspect);

        if (animateSubject) {
            const float angle = subjectTime * 0.35f;
            const glm::vec3 target(0.2f * std::sin(angle * 1.7f), 0.95f, -0.2f);
            const glm::vec3 pos(-2.8f + 0.45f * std::sin(angle), 1.35f,
                                3.0f + 0.35f * std::cos(angle));
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
