///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _BASE_PANEL_HPP_
#define _BASE_PANEL_HPP_
#include "panel.hpp"

namespace KE {

class App;

class PerformancePanel : public Panel {
  private:
  public:
    PerformancePanel();
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

class MenuBarPanel : public Panel {
  private:
    App* _app;

  public:
    MenuBarPanel(App* app);
    ~MenuBarPanel();
    virtual void buildPanel();
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
