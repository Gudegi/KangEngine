///
/// SkeletonVisualBridge — renderer-backed line/sphere skeleton visualization.
///

#ifndef _SKELETON_VISUAL_BRIDGE_HPP_
#define _SKELETON_VISUAL_BRIDGE_HPP_

#include "animation/skeleton_motion.hpp"
#include "animation/skeleton_state.hpp"
#include "engine/graphics/material/colors.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include <glm/vec4.hpp>
#include <optional>
#include <string>

namespace KE {

class App;
namespace Backend {
class Shader;
}

namespace Bridge {

inline glm::vec4 skeletonVisualColor(ColorType type, float alpha = 1.0f) {
    const Color& color = ColorLibrary::get(type);
    return glm::vec4(color.r, color.g, color.b, alpha);
}

struct SkeletonVisualConfig {
    glm::vec4 boneColor = skeletonVisualColor(ColorType::PASTEL_SKY);
    glm::vec4 jointColor = skeletonVisualColor(ColorType::PASTEL_CORAL);
    float boneRadius = 0.006f;
    float jointRadius = 0.025f;
    int segments = 8;
    bool visible = true;
    bool showJoints = true;
};

class SkeletonVisualBridge {
  public:
    SkeletonVisualBridge() = default;

    static SkeletonVisualBridge define(App* app, Backend::Shader* shader,
                                       const std::string& basePath,
                                       const Animation::SkeletonState& state,
                                       const SkeletonVisualConfig& config = {});

    static SkeletonVisualBridge define(App* app, Backend::Shader* shader,
                                       const std::string& basePath,
                                       const Animation::SkeletonMotion& motion,
                                       float time, bool loop = true,
                                       const SkeletonVisualConfig& config = {});

    void applyState(const Animation::SkeletonState& state);
    void applyMotion(const Animation::SkeletonMotion& motion, float time,
                     bool loop = true);
    void setVisible(bool visible);
    void setShowJoints(bool showJoints);
    const SkeletonVisualConfig& config() const { return _config; }

    MeshHandle boneHandle() const { return _boneHandle; }
    MeshHandle jointHandle() const { return _jointHandle; }

  private:
    App* _app = nullptr;                // non-owning
    Backend::Shader* _shader = nullptr; // non-owning
    std::string _basePath;
    SkeletonVisualConfig _config;
    std::optional<Animation::SkeletonState> _lastState;
    MeshHandle _boneHandle = InvalidHandle;
    MeshHandle _jointHandle = InvalidHandle;
};

} // namespace Bridge
} // namespace KE

#endif
