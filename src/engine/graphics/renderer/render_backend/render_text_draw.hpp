#pragma once

// Persistent screen and world text commands for render backends.

#include "render_text_data.hpp"

namespace KE {

class RenderTextDraw {
  public:
    virtual ~RenderTextDraw() = default;
    virtual void setWorldText(const std::string& path,
                              const WorldTextDesc& desc) = 0;
    virtual void setWorldString(const std::string& path, std::string text) = 0;
    virtual void setWorldPosition(const std::string& path,
                                  const glm::vec3& position) = 0;
    virtual void setWorldHidden(const std::string& path, bool hidden) = 0;
    virtual void removeWorldText(const std::string& path) = 0;
    virtual void clearWorldText() = 0;
    virtual void setScreenText(const std::string& path,
                               const ScreenTextDesc& desc) = 0;
    virtual void setScreenString(const std::string& path, std::string text) = 0;
    virtual void setScreenPosition(const std::string& path,
                                   const glm::vec2& position) = 0;
    virtual void setScreenHidden(const std::string& path, bool hidden) = 0;
    virtual void removeScreenText(const std::string& path) = 0;
    virtual void clearScreenText() = 0;
};

} // namespace KE
