#include "kangEngine.hpp"
#include <ImGuizmo.h>
#include <imgui.h>
#include <iostream>
#include <stdexcept>

using namespace KE;

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

class SelectionApp : public App {
  public:
    int picks = 0;
    int forceBegins = 0;
    int forceEnds = 0;
    void onRayPicked(const RayPickResult&) override { ++picks; }
    void onForceDragBegin(const RayPickResult&, const glm::vec3&) override {
        ++forceBegins;
    }
    void onForceDragEnd() override { ++forceEnds; }
};
} // namespace

int main() {
    SelectionApp app;
    app.initialize(640, 480, true, UpAxis::Y, Backend::BackendType::OpenGL,
                   Scene::BackendType::Native, true);
    ColorMaterial material;
    auto* prim = app.getScene()->definePrim("/cube", Scene::PrimType::Mesh);
    prim->setMeshData(
        std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(1)));
    app.addRenderable(&material, prim);
    auto resetCamera = [&] {
        app.getCamera().setCameraPos({0, 0, 6});
        app.getCamera().setTargetPos({0, 0, 0});
    };
    resetCamera();
    app.drawFrame();
    auto move = [&](double x, double y) {
        glfwSetCursorPos(app.getWindow(), x, y);
        app.cursorPositionCallback(app.getWindow(), x, y);
    };
    auto press = [&](int mods = 0) {
        app.mouseButtonCallback(app.getWindow(), GLFW_MOUSE_BUTTON_LEFT,
                                GLFW_PRESS, mods);
    };
    auto release = [&] {
        app.mouseButtonCallback(app.getWindow(), GLFW_MOUSE_BUTTON_LEFT,
                                GLFW_RELEASE, 0);
    };
    auto click = [&](double x, double y) {
        move(x, y);
        press();
        release();
        app.drawFrame();
    };

    for (auto mode : {InteractionMode::Inspect, InteractionMode::Edit,
                      InteractionMode::Force}) {
        app.setInteractionMode(mode);
        app.clearSelection();
        move(320, 240);
        press();
        app.drawFrame();
        require(!app.hasSelection(), "selection must wait for release");
        release();
        app.drawFrame();
        require(app.getSelection().prim == prim, "plain click did not select");
        require(app.forceBegins == 0, "plain click started a force drag");
        click(10, 10);
        require(!app.hasSelection(),
                "background click did not clear selection");
    }

    click(320, 240);
    const int beforeDrag = app.picks;
    const auto cameraPosition = app.getCamera().getCameraPos();
    move(320, 240);
    press();
    move(322, 241);
    app.processInput();
    require(app.getCamera().getCameraPos() == cameraPosition,
            "click jitter moved the camera");
    move(380, 240);
    app.processInput();
    require(app.getCamera().getCameraPos() != cameraPosition,
            "left drag no longer orbits the camera");
    move(320, 240);
    release();
    app.drawFrame();
    require(app.picks == beforeDrag && app.getSelection().prim == prim,
            "returning a camera drag to its origin generated a click");
    resetCamera();
    app.drawFrame();

    // Capture at press must survive releasing outside the UI.
    app._hideUI = false;
    app.setUILayoutMode(UILayoutMode::Overlay);
    ImGui::GetIO().WantCaptureMouse = true;
    move(320, 240);
    press();
    ImGui::GetIO().WantCaptureMouse = false;
    move(10, 10);
    release();
    app._hideUI = true;
    app.drawFrame();
    require(app.picks == beforeDrag && app.hasSelection(),
            "UI gesture reached scene selection");

    app.setInteractionMode(InteractionMode::Force);
    move(320, 240);
    press(GLFW_MOD_SHIFT);
    app.drawFrame();
    require(app.forceBegins == 1, "Shift press did not begin force dragging");
    release();
    require(app.forceEnds == 1, "release did not end force dragging");
    press(GLFW_MOD_SHIFT);
    release();
    app.drawFrame();
    require(app.forceBegins == 1 && app.forceEnds == 1,
            "quick Shift click left a force drag active");
    app.setInteractionMode(InteractionMode::Edit);
    const int beforeShift = app.picks;
    press(GLFW_MOD_SHIFT);
    release();
    app.drawFrame();
    require(app.picks == beforeShift, "Shift click selected in Edit mode");

    // Exercise the real gizmo, including dragging away from its initial handle.
    app._hideUI = false;
    app.setUILayoutMode(UILayoutMode::Overlay);
    for (const char* name : {"Scene", "Inspector", "Renderer Debug",
                             "Performance", "Viewport", "Camera View"})
        app._panelManager.setPanelOpen(name, false);
    app.setGizmoOperation(GizmoOperation::Translate);
    ImGui::GetIO().IniFilename = nullptr;
    auto uiMove = [&](float x, float y) {
        move(x, y);
        ImGui::GetIO().AddMousePosEvent(x, y);
        app.drawFrame();
    };
    app.selectPrim(prim);
    uiMove(320, 240);
    uiMove(320, 240);
    require(ImGuizmo::IsOver(), "test cursor missed the translation gizmo");
    const int beforeGizmo = app.picks;
    const auto beforeGizmoCamera = app.getCamera().getCameraPos();
    const auto beforeTransform = prim->computeModelMatrix();
    press();
    ImGui::GetIO().AddMouseButtonEvent(0, true);
    app.drawFrame();
    require(ImGuizmo::IsUsing(), "gizmo did not capture the mouse");
    uiMove(350, 260);
    app.processInput();
    require(prim->computeModelMatrix() != beforeTransform,
            "gizmo drag did not transform the selected prim");
    require(app.getCamera().getCameraPos() == beforeGizmoCamera,
            "gizmo drag moved the camera");
    release();
    ImGui::GetIO().AddMouseButtonEvent(0, false);
    app.drawFrame();
    require(app.picks == beforeGizmo && app.getSelection().prim == prim,
            "gizmo release changed the scene selection");
    app.close();
    std::cout << "PASS: plain selection, deselection, camera/UI/gizmo capture, "
                 "and explicit force dragging\n";
}
