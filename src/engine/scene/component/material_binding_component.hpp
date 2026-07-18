#ifndef _SCENE_MATERIAL_BINDING_COMPONENT_HPP_
#define _SCENE_MATERIAL_BINDING_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"

namespace KE {

class Material;

namespace Scene {

class Prim;

// Scene-level material relationship for a Prim.
//
// MaterialBindingComponent does not own the material yet. The current Python
// examples keep material lifetimes explicitly, and the renderer batches by
// Material* already. A future material registry / shared_ptr migration can
// replace this pointer without changing the scene concept.
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
