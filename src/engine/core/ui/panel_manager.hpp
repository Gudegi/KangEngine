///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _PANEL_MANAGER_HPP_
#define _PANEL_MANAGER_HPP_

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <list>
#include <memory>
#include <string>
#include "panel.hpp"

namespace KE {

enum class UILayoutMode {
    Viewer,  // Full-window scene view + docked panels
    Editor,  // Scene view in an interactive viewport panel.
    Overlay, // Full-window scene view + floating panels
};

class PanelManager {

    ///
    /// @brief Define panel manager
    ///

  public:
    static constexpr const char* PANEL_SCENE = "Scene";
    static constexpr const char* PANEL_RENDERER_DEBUG = "Renderer Debug";
    static constexpr const char* PANEL_PERFORMANCE = "Performance";
    static constexpr const char* PANEL_INSPECTOR = "Inspector";
    static constexpr const char* PANEL_VIEWPORT = "Viewport";
    static constexpr const char* PANEL_CAMERA_VIEW = "Camera View";

  private:
    float _fontSize = 12.0f;
    UILayoutMode _layoutMode = UILayoutMode::Viewer;
    bool _layoutInitialized = false;

    ///
    /// @brief list contains the each panels.
    /// @todo memory leak protection is need.
    ///
    // std::list<Panel*> _panels;
    std::list<std::unique_ptr<Panel>> _panels;

    ///
    /// @brief Scale imgui contents correspondings to glfw monitor size.
    ///
    void scaleDPI(GLFWwindow* window);

    ///
    /// @brief Set imgui theme as dracula theme.
    ///
    void draculaTheme();

    ///
    /// @brief Get DPI scale factor from window
    ///
    float getDPIScale(GLFWwindow* window);

    ///
    /// @brief Build the initial dock layout.
    ///
    void initLayout(ImGuiID dockspace_id);

    Panel* findPanel(const char* name);
    const Panel* findPanel(const char* name) const;

  public:
    ///
    /// @brief Construct the manager that initializes imgui, scales, and, sets
    /// theme.
    ///
    PanelManager();

    ///
    /// @brief Destruct panel manager
    /// @todo check it is ok that when you terminate imgui.
    ///
    ~PanelManager();

    void init(GLFWwindow* window);

    float getFontSize() { return _fontSize; }

    void setFontSize(float fontSize) { _fontSize = fontSize; }

    ///
    /// @brief Add a panel to render info.
    ///
    // void addPanel(Panel* panel);
    void addPanel(std::unique_ptr<Panel> panel);
    UILayoutMode getLayoutMode() const { return _layoutMode; }
    void setLayoutMode(UILayoutMode mode);
    bool isPanelOpen(const char* name) const;
    void setPanelOpen(const char* name, bool open);
    void resetLayout();

    ///
    /// @brief prepare new frames. This should be called on top of main render
    /// loop.
    ///
    void preRender();

    ///
    /// @brief actual render function. This should be called in main render
    /// loop.
    ///
    void render();

    ///
    /// @brief finalize imgui's render function. This should be called on bottom
    /// of main render loop.
    ///
    void postRender();

    ///
    /// @brief Load a font from file path with specified size.
    /// @param fontPath Path to the font file (TTF format)
    /// @param fontSize Font size in pixels
    /// @param loadKorean Whether to load Korean glyph ranges (default: true)
    /// @return true if font loaded successfully, false otherwise
    ///
    bool loadFont(const std::string& fontPath, float fontSize,
                  bool loadKorean = true);
    bool loadFont(const std::string& fontPath, bool loadKorean = true);
    bool mergeIconFont(const std::string& fontPath);
};

} // namespace KE

#endif
