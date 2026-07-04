#ifndef _SCENE_RENDER_COMPONENT_HPP_
#define _SCENE_RENDER_COMPONENT_HPP_

#include "engine/graphics/renderer/renderer_types.hpp"

#include <cstdint>
#include <functional>
#include <memory>

namespace KE {
namespace Scene {

struct MeshData;
class Prim;
class SceneRenderSystem;

// Renderer-independent visual state attached to one scene Prim.
// Renderer registration and RenderableHandle ownership belong to
// SceneRenderSystem, not to this component.
class RenderComponent {
  public:
    RenderComponent(const RenderComponent&) = delete;
    RenderComponent& operator=(const RenderComponent&) = delete;

    bool isAttached() const { return _owner != nullptr; }
    Prim* owner() const { return _owner; }

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
    uint64_t version() const { return _version; }

  private:
    friend class Prim;
    friend class SceneRenderSystem;

    explicit RenderComponent(Prim* owner);
    void detach();
    void setRegistrationCallbacks(
        std::function<void(RenderComponent&)> detachCallback,
        std::function<void(RenderComponent&)> changeCallback);
    void clearRegistrationCallbacks();
    void requireAttached() const;
    void markChanged();

    Prim* _owner = nullptr;
    bool _doubleSided = false;
    bool _castsShadow = true;
    AlphaMode _alphaMode = AlphaMode::Opaque;
    float _alphaCutoff = 0.5f;
    TransformSource _transformSource = TransformSource::SceneGraph;
    uint64_t _version = 1;
    bool _registered = false;
    std::function<void(RenderComponent&)> _detachCallback;
    std::function<void(RenderComponent&)> _changeCallback;
};

} // namespace Scene
} // namespace KE

#endif
