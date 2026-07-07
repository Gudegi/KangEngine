#include "panel_manager.hpp"
#include <IconsFontAwesome7.h>
#include <ImGuizmo.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <imgui_internal.h>

namespace KE {

PanelManager::PanelManager() {}

PanelManager::~PanelManager() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void PanelManager::init(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |=
        ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable
    // Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // ImGui::StyleColorsDark();
    // ImGui::StyleColorsClassic();
    draculaTheme();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 410");
    scaleDPI(window);

    // Load default font
    float dpiScale = getDPIScale(window);
#ifdef __APPLE__
    // TODO: improve me.
    _fontSize = 13.0f;
#else
    _fontSize = 13.0f; //  * dpiScale;
#endif
    // loadFont("./assets/fonts/godoFont/GodoM.ttf", fontSize, true);
}

float PanelManager::getDPIScale(GLFWwindow* window) {
    float xScale, yScale;
    GLFWmonitor* monitor = glfwGetWindowMonitor(window);
    if (!monitor) {
        monitor = glfwGetPrimaryMonitor();
    }
    glfwGetMonitorContentScale(monitor, &xScale, &yScale);
    return (xScale + yScale) * 0.5f;
}

void PanelManager::scaleDPI(GLFWwindow* window) {
    float dpiScale = getDPIScale(window);

#ifdef __linux__
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(dpiScale);
    io.FontGlobalScale = dpiScale;
#endif
}

bool PanelManager::loadFont(const std::string& fontPath, bool loadKorean) {
    return loadFont(fontPath, _fontSize, loadKorean);
}

bool PanelManager::loadFont(const std::string& fontPath, float fontSize,
                            bool loadKorean) {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();

    ImFontConfig config;
    config.OversampleH = 2;
    config.OversampleV = 2;

    ImFont* font = nullptr;
    if (loadKorean) {
        font = io.Fonts->AddFontFromFileTTF(fontPath.c_str(), fontSize, &config,
                                            io.Fonts->GetGlyphRangesKorean());
    } else {
        font =
            io.Fonts->AddFontFromFileTTF(fontPath.c_str(), fontSize, &config);
    }

    if (!font) {
        std::cerr << "Failed to load font: " << fontPath
                  << ", using default font" << std::endl;
        io.Fonts->AddFontDefault();
        io.Fonts->Build();
        return false;
    }

    io.Fonts->Build();
    return true;
}

bool PanelManager::mergeIconFont(const std::string& fontPath) {
    ImGuiIO& io = ImGui::GetIO();
    const float iconFontSize = _fontSize * 2.0f / 3.0f;
    ImFontConfig config;
    config.MergeMode = true;
    config.PixelSnapH = true;
    config.GlyphMinAdvanceX = iconFontSize;
    static const ImWchar iconRanges[] = {ICON_MIN_FA, ICON_MAX_16_FA, 0};

    ImFont* font = io.Fonts->AddFontFromFileTTF(fontPath.c_str(), iconFontSize,
                                                &config, iconRanges);
    if (!font) {
        std::cerr << "Failed to load icon font: " << fontPath << std::endl;
        return false;
    }

    io.Fonts->Build();
    return true;
}

// void PanelManager::addPanel(Panel* panel)
void PanelManager::addPanel(std::unique_ptr<Panel> panel) {
    _panels.push_back(std::move(panel));
}

void PanelManager::setLayoutMode(UILayoutMode mode) {
    if (_layoutMode == mode)
        return;
    _layoutMode = mode;
    setPanelOpen(PANEL_VIEWPORT, _layoutMode == UILayoutMode::Editor);
    setPanelOpen(PANEL_CAMERA_VIEW, false);
    resetLayout();
}

Panel* PanelManager::findPanel(const char* name) {
    for (auto& panel : _panels) {
        if (panel->name() == name)
            return panel.get();
    }
    return nullptr;
}

const Panel* PanelManager::findPanel(const char* name) const {
    for (const auto& panel : _panels) {
        if (panel->name() == name)
            return panel.get();
    }
    return nullptr;
}

bool PanelManager::isPanelOpen(const char* name) const {
    const Panel* panel = findPanel(name);
    return panel ? panel->isOpen() : false;
}

void PanelManager::setPanelOpen(const char* name, bool open) {
    if (Panel* panel = findPanel(name)) {
        const bool changed = panel->isOpen() != open;
        panel->setOpen(open);
        if (changed && _layoutMode == UILayoutMode::Editor &&
            (panel->name() == PANEL_CAMERA_VIEW ||
             panel->name() == PANEL_VIEWPORT)) {
            resetLayout();
        }
    }
}

void PanelManager::resetLayout() { _layoutInitialized = false; }

void PanelManager::preRender() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();

    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
        ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
        ImGui::DockSpaceOverViewport(dockspace_id, ImGui::GetMainViewport(),
                                     ImGuiDockNodeFlags_PassthruCentralNode);

        if (!_layoutInitialized) {
            initLayout(dockspace_id);
            _layoutInitialized = true;
        }
    }
}

void PanelManager::initLayout(ImGuiID dockspace_id) {
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

    ImGuiID dock_main_id = dockspace_id;

    if (_layoutMode == UILayoutMode::Overlay) {
        ImGui::DockBuilderFinish(dockspace_id);
        return;
    }

    ImGuiID dock_id_right = ImGui::DockBuilderSplitNode(
        dock_main_id, ImGuiDir_Right, 0.24f, nullptr, &dock_main_id);
    ImGuiID dock_id_scene = dock_id_right;
    ImGuiID dock_id_debug = ImGui::DockBuilderSplitNode(
        dock_id_scene, ImGuiDir_Down, 0.58f, nullptr, &dock_id_scene);

    ImGui::DockBuilderDockWindow(PANEL_SCENE, dock_id_scene);
    ImGui::DockBuilderDockWindow(PANEL_RENDERER_DEBUG, dock_id_debug);
    ImGui::DockBuilderDockWindow(PANEL_PERFORMANCE, dock_id_debug);
    ImGui::DockBuilderDockWindow(PANEL_INSPECTOR, dock_id_debug);
    if (_layoutMode == UILayoutMode::Editor) {
        if (isPanelOpen(PANEL_CAMERA_VIEW)) {
            ImGuiID dock_id_camera_view = ImGui::DockBuilderSplitNode(
                dock_main_id, ImGuiDir_Down, 0.32f, nullptr, &dock_main_id);
            ImGui::DockBuilderDockWindow(PANEL_VIEWPORT, dock_main_id);
            ImGui::DockBuilderDockWindow(PANEL_CAMERA_VIEW,
                                         dock_id_camera_view);
        } else {
            ImGui::DockBuilderDockWindow(PANEL_VIEWPORT, dock_main_id);
        }
    }
    ImGui::DockBuilderFinish(dockspace_id);
}

void PanelManager::render() {
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        // Your input functions here
    }

    for (auto& panel : _panels) {
        if (!panel->isOpen())
            continue;
        if (panel->name() == PANEL_VIEWPORT &&
            _layoutMode != UILayoutMode::Editor)
            continue;
        if (panel->name() == PANEL_CAMERA_VIEW &&
            _layoutMode != UILayoutMode::Editor)
            continue;
        panel->buildPanel();
    }
}

void PanelManager::postRender() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void PanelManager::draculaTheme() {
    // reference :
    // https://github.com/ocornut/imgui/issues/707#issuecomment-1372640066
    auto& colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_WindowBg] = ImVec4{0.1f, 0.1f, 0.13f, 1.0f};
    colors[ImGuiCol_MenuBarBg] = ImVec4{0.16f, 0.16f, 0.21f, 1.0f};

    // Border
    colors[ImGuiCol_Border] = ImVec4{0.44f, 0.37f, 0.61f, 0.29f};
    colors[ImGuiCol_BorderShadow] = ImVec4{0.0f, 0.0f, 0.0f, 0.24f};

    // Text
    colors[ImGuiCol_Text] = ImVec4{1.0f, 1.0f, 1.0f, 1.0f};
    colors[ImGuiCol_TextDisabled] = ImVec4{0.5f, 0.5f, 0.5f, 1.0f};

    // Headers
    colors[ImGuiCol_Header] = ImVec4{0.13f, 0.13f, 0.17, 1.0f};
    colors[ImGuiCol_HeaderHovered] = ImVec4{0.19f, 0.2f, 0.25f, 1.0f};
    colors[ImGuiCol_HeaderActive] = ImVec4{0.16f, 0.16f, 0.21f, 1.0f};

    // Buttons
    colors[ImGuiCol_Button] = ImVec4{0.13f, 0.13f, 0.17, 1.0f};
    colors[ImGuiCol_ButtonHovered] = ImVec4{0.19f, 0.2f, 0.25f, 1.0f};
    colors[ImGuiCol_ButtonActive] = ImVec4{0.16f, 0.16f, 0.21f, 1.0f};
    colors[ImGuiCol_CheckMark] = ImVec4{0.74f, 0.58f, 0.98f, 1.0f};

    // Popups
    colors[ImGuiCol_PopupBg] = ImVec4{0.1f, 0.1f, 0.13f, 0.92f};

    // Slider
    colors[ImGuiCol_SliderGrab] = ImVec4{0.44f, 0.37f, 0.61f, 0.54f};
    colors[ImGuiCol_SliderGrabActive] = ImVec4{0.74f, 0.58f, 0.98f, 0.54f};

    // Frame BG
    colors[ImGuiCol_FrameBg] = ImVec4{0.13f, 0.13, 0.17, 1.0f};
    colors[ImGuiCol_FrameBgHovered] = ImVec4{0.19f, 0.2f, 0.25f, 1.0f};
    colors[ImGuiCol_FrameBgActive] = ImVec4{0.16f, 0.16f, 0.21f, 1.0f};

    // Tabs
    colors[ImGuiCol_Tab] = ImVec4{0.16f, 0.16f, 0.21f, 1.0f};
    colors[ImGuiCol_TabHovered] = ImVec4{0.24, 0.24f, 0.32f, 1.0f};
    colors[ImGuiCol_TabActive] = ImVec4{0.2f, 0.22f, 0.27f, 1.0f};
    colors[ImGuiCol_TabUnfocused] = ImVec4{0.16f, 0.16f, 0.21f, 1.0f};
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4{0.16f, 0.16f, 0.21f, 1.0f};

    // Title
    colors[ImGuiCol_TitleBg] = ImVec4{0.16f, 0.16f, 0.21f, 1.0f};
    colors[ImGuiCol_TitleBgActive] = ImVec4{0.16f, 0.16f, 0.21f, 1.0f};
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4{0.16f, 0.16f, 0.21f, 1.0f};

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg] = ImVec4{0.1f, 0.1f, 0.13f, 1.0f};
    colors[ImGuiCol_ScrollbarGrab] = ImVec4{0.16f, 0.16f, 0.21f, 1.0f};
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4{0.19f, 0.2f, 0.25f, 1.0f};
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4{0.24f, 0.24f, 0.32f, 1.0f};

    // Seperator
    colors[ImGuiCol_Separator] = ImVec4{0.44f, 0.37f, 0.61f, 1.0f};
    colors[ImGuiCol_SeparatorHovered] = ImVec4{0.74f, 0.58f, 0.98f, 1.0f};
    colors[ImGuiCol_SeparatorActive] = ImVec4{0.84f, 0.58f, 1.0f, 1.0f};

    // Resize Grip
    colors[ImGuiCol_ResizeGrip] = ImVec4{0.44f, 0.37f, 0.61f, 0.29f};
    colors[ImGuiCol_ResizeGripHovered] = ImVec4{0.74f, 0.58f, 0.98f, 0.29f};
    colors[ImGuiCol_ResizeGripActive] = ImVec4{0.84f, 0.58f, 1.0f, 0.29f};

    // Docking
    colors[ImGuiCol_DockingPreview] = ImVec4{0.44f, 0.37f, 0.61f, 1.0f};

    auto& style = ImGui::GetStyle();
    style.TabRounding = 4;
    style.ScrollbarRounding = 9;
    style.WindowRounding = 7;
    style.GrabRounding = 3;
    style.FrameRounding = 3;
    style.PopupRounding = 4;
    style.ChildRounding = 4;
}

} // namespace KE
