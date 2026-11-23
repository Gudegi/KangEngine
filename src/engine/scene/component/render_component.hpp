#ifndef _SCENE_RENDER_COMPONENT_HPP_
#define _SCENE_RENDER_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"

#include <functional>
#include <memory>
#include <optional>

namespace KE {
namespace Scene {

struct MeshData;
class Prim;
class MeshComponent;
class SceneRenderSystem;

// Owns a Prim's render settings and display color; visibility is stored in Prim.
// SceneRenderSystem manages registration and RenderableHandles.
class RenderComponent : public ComponentBase {
  public:
    bool isVisible() const;
    void setVisible(bool visible);

    std::optional<glm::vec4> displayColor() const;
    void setDisplayColor(glm::vec4 color);

    bool isDoubleSided() const;
    void setDoubleSided(bool doubleSided);

    bool castsShadow() const;
    void setCastsShadow(bool castsShadow);

    AlphaMode alphaMode() const;
    float alphaCutoff() const;
    void setAlphaMode(AlphaMode mode, float cutoff = 0.5f);

    TransformSource transformSource() const;
    void setTransformSource(TransformSource source);

    std::shared_ptr<MeshData> resolveMeshData() const;

  private:
    friend class Prim;
    friend class MeshComponent;
    friend class MaterialBindingComponent;
    friend class SceneRenderSystem;

    explicit RenderComponent(Prim* owner);
    void detach();
    void setRegistrationCallbacks(
        std::function<void(RenderComponent&)> detachCallback,
        std::function<void(RenderComponent&)> changeCallback);
    void clearRegistrationCallbacks();
    void markChanged();

    // Linear RGBA multiplier; unauthored colors act as white.
    std::optional<glm::vec4> _displayColor;
    bool _doubleSided = false;
    bool _castsShadow = true;
    AlphaMode _alphaMode = AlphaMode::Opaque;
    float _alphaCutoff = 0.5f;
    TransformSource _transformSource = TransformSource::SceneGraph;
    bool _registered = false;
    std::function<void(RenderComponent&)> _detachCallback;
    std::function<void(RenderComponent&)> _changeCallback;
};

} // namespace Scene
} // namespace KE

#endif
