///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _BASE_PANEL_HPP_
#define _BASE_PANEL_HPP_
#include "imgui.h"
#include "panel.hpp"
#include <string>

namespace KE {

class App;

class PerformancePanel : public Panel {
  private:
    App* _app;

  public:
    PerformancePanel(App* app = nullptr);
    ~PerformancePanel();
    virtual void buildPanel();
};

class RendererDebugPanel : public Panel {
  private:
    App* _app;

  public:
    RendererDebugPanel(App* app);
    ~RendererDebugPanel();
    virtual void buildPanel();
};

class InspectorPanel : public Panel {
  private:
    App* _app;

  public:
    explicit InspectorPanel(App* app);
    ~InspectorPanel();
    virtual void buildPanel();
};

class MenuBarPanel : public Panel {
  private:
    App* _app;

  public:
    MenuBarPanel(App* app);
    ~MenuBarPanel();
    virtual void buildPanel();
};

class ViewportPanel : public Panel {
  private:
    App* _app = nullptr;
    std::string _cameraLabel;
    ImVec2 _contentMin = {0.0f, 0.0f};
    ImVec2 _contentSize = {0.0f, 0.0f};
    ImVec2 _imageMin = {0.0f, 0.0f};
    ImVec2 _imageSize = {0.0f, 0.0f};
    bool _hovered = false;
    bool _focused = false;

  public:
    explicit ViewportPanel(App* app, std::string name = "Viewport",
                           std::string cameraLabel = "Main Camera");
    ~ViewportPanel();
    virtual void buildPanel();
    const std::string& cameraLabel() const { return _cameraLabel; }
    void setCameraLabel(std::string cameraLabel);
    ImVec2 contentMin() const { return _contentMin; }
    ImVec2 contentSize() const { return _contentSize; }
    ImVec2 imageMin() const { return _imageMin; }
    ImVec2 imageSize() const { return _imageSize; }
    bool isHovered() const { return _hovered; }
    bool isFocused() const { return _focused; }
};

class ScenePanel : public Panel {
  private:
    App* _app;

  public:
    ScenePanel(App* app);
    ~ScenePanel();
    virtual void buildPanel();
};

} // namespace KE

#endif
