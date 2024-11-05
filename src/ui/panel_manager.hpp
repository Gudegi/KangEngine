#ifndef _PANEL_MANAGER_HPP_
#define _PANEL_MANAGER_HPP_

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <list>
#include "panel.hpp"
#include <memory>

class PanelManager{

///
/// @brief Define panel manager 
///

private:

    ///
    /// @brief list contains the each panels.
    /// @todo memory leak protection is need.
    ///
    std::list<Panel*> m_panels;
    //std::list<std::unique_ptr<Panel>> m_panels;

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
    PanelManager(GLFWwindow* window);

    ///
    /// @brief Destruct panel manager
    /// @todo check it is ok that when you terminate imgui.
    /// 
    ~PanelManager();

    ///
    /// @brief Add a panel to render info. 
    /// 
    void addPanel(Panel* panel);
    //void addPanel(std::unique_ptr<Panel> panel);

    ///
    /// @brief Render function. This should be called in main render loop.
    /// 
    void render();

};

#endif