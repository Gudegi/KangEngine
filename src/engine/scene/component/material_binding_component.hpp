#ifndef _SCENE_MATERIAL_BINDING_COMPONENT_HPP_
#define _SCENE_MATERIAL_BINDING_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"

namespace KE {

class Material;

namespace Scene {

class Prim;

// References a Prim's Material without owning it or copying its values.
// The caller must keep the Material alive while it is bound.
class MaterialBindingComponent : public ComponentBase {
  public:
    Material* material() const;
    void setMaterial(Material* material);
    void clearMaterial();

  private:
    friend class Prim;

    explicit MaterialBindingComponent(Prim* owner);
    void detach();

    Material* _material = nullptr;
};

} // namespace Scene
} // namespace KE

#endif
