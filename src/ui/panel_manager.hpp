///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _PANEL_MANAGER_HPP_
#define _PANEL_MANAGER_HPP_

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <list>
#include "panel.hpp"
#include <memory>

class PanelManager
{

///
/// @brief Define panel manager 
///

private:

    ///
    /// @brief list contains the each panels.
    /// @todo memory leak protection is need.
    ///
    //std::list<Panel*> _panels;
    std::list<std::unique_ptr<Panel>> _panels;

    ///
    /// @brief Scale imgui contents correspondings to glfw monitor size.
    ///
    void scaleDPI(GLFWwindow* window);

    ///
    /// @brief Set imgui theme as dracula theme.
    ///
    void draculaTheme();

public:

    ///
    /// @brief Construct the manager that initializes imgui, scales, and, sets theme.
    ///
    PanelManager();

    ///
    /// @brief Destruct panel manager
    /// @todo check it is ok that when you terminate imgui.
    /// 
    ~PanelManager();

    void init(GLFWwindow* window);

    ///
    /// @brief Add a panel to render info. 
    /// 
    //void addPanel(Panel* panel);
    void addPanel(std::unique_ptr<Panel> panel);

    ///
    /// @brief prepare new frames. This should be called on top of main render loop.
    /// 
    void preRender();

    ///
    /// @brief actual render function. This should be called in main render loop.
    /// 
    void render();

    ///
    /// @brief finalize imgui's render function. This should be called on bottom of main render loop.
    /// 
    void postRender();

};

#endif