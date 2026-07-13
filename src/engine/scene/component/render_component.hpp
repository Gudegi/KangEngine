#ifndef _SCENE_RENDER_COMPONENT_HPP_
#define _SCENE_RENDER_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"

#include <functional>
#include <memory>

namespace KE {
namespace Scene {

struct MeshData;
class Prim;
class MeshComponent;
class SceneRenderSystem;

// Renderer-independent visual state attached to one scene Prim.
// Renderer registration and RenderableHandle ownership belong to
// SceneRenderSystem, not to this component.
class RenderComponent : public ComponentBase {
  public:
    bool isVisible() const;
    void setVisible(bool visible);

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
    friend class SceneRenderSystem;

    explicit RenderComponent(Prim* owner);
    void detach();
    void setRegistrationCallbacks(
        std::function<void(RenderComponent&)> detachCallback,
        std::function<void(RenderComponent&)> changeCallback);
    void clearRegistrationCallbacks();
    void markChanged();

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
