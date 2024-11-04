#ifndef _PANEL_MANAGER_HPP_
#define _PANEL_MANAGER_HPP_

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <list>
#include "panel.hpp"
#include <memory>

class PanelManager{

/*
    Define panel manager 
*/

private:
    std::list<Panel*> m_panels;
    //std::list<std::unique_ptr<Panel>> m_panels;
    void scaleDPI(GLFWwindow* window);
    void draculaTheme();

public:
    PanelManager(GLFWwindow* window);
    ~PanelManager();
    void addPanel(Panel* panel);
    //void addPanel(std::unique_ptr<Panel> panel);
    void render();

};

#endif