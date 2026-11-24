#pragma once

// Persistent world-space debug batches consumed by a render backend.

#include "engine/graphics/material/colors.hpp"
#include <glm/glm.hpp>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace KE {

class RenderDebugDraw {
  public:
    virtual ~RenderDebugDraw() = default;

    // Sizes are pixels. Overlay batches bypass scene depth testing.
    void logLines(const std::string& path, const std::vector<glm::vec3>& starts,
                  const std::vector<glm::vec3>& ends,
                  const std::vector<glm::vec4>& colors = {}, float width = 1,
                  bool hidden = false, bool overlay = false) {
        if (path.empty())
            return;
        if (starts.size() != ends.size())
            throw std::invalid_argument(
                "Debug lines require matching endpoints");
        validateColors(colors, starts.size());

        setLines(path, starts, ends, colors, std::max(1.0f, width), hidden,
                 overlay);
    }

    void logPoints(const std::string& path,
                   const std::vector<glm::vec3>& points,
                   const std::vector<glm::vec4>& colors = {}, float size = 6,
                   bool hidden = false, bool overlay = false) {
        if (path.empty())
            return;
        validateColors(colors, points.size());

        setPoints(path, points, colors, std::max(1.0f, size), hidden, overlay);
    }

    void logAxes(const std::string& path, const glm::mat4& transform,
                 float length = 1, float width = 1, bool hidden = false,
                 bool overlay = false) {
        logAxes(path, glm::vec3(transform[3]), glm::vec3(transform[0]),
                glm::vec3(transform[1]), glm::vec3(transform[2]), length, width,
                hidden, overlay);
    }

    void logAxes(const std::string& path, const glm::vec3& origin,
                 const glm::vec3& xAxis, const glm::vec3& yAxis,
                 const glm::vec3& zAxis, float length = 1, float width = 1,
                 bool hidden = false, bool overlay = false) {
        const float safeLength = std::max(0.0f, length);
        logLines(path, {origin, origin, origin},
                 {origin + glm::normalize(xAxis) * safeLength,
                  origin + glm::normalize(yAxis) * safeLength,
                  origin + glm::normalize(zAxis) * safeLength},
                 {ColorLibrary::getVec4(ColorType::RED),
                  ColorLibrary::getVec4(ColorType::GREEN),
                  ColorLibrary::getVec4(ColorType::BLUE)},
                 width, hidden, overlay);
    }

    virtual void clearLines(const std::string& path) = 0;
    virtual void clearPoints(const std::string& path) = 0;
    virtual void setVisible(const std::string& path, bool visible) = 0;
    virtual void remove(const std::string& path) = 0;
    virtual void removeAll() = 0;

  protected:
    virtual void setLines(const std::string&, const std::vector<glm::vec3>&,
                          const std::vector<glm::vec3>&,
                          const std::vector<glm::vec4>&, float, bool, bool) = 0;
    virtual void setPoints(const std::string&, const std::vector<glm::vec3>&,
                           const std::vector<glm::vec4>&, float, bool,
                           bool) = 0;

  private:
    static void validateColors(const std::vector<glm::vec4>& colors,
                               size_t count) {
        if (!colors.empty() && colors.size() != 1 && colors.size() != count)
            throw std::invalid_argument("Debug colors must be empty, length 1, "
                                        "or match the primitive count");
    }
};

} // namespace KE
