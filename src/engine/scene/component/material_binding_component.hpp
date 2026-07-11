#ifndef _SCENE_MATERIAL_BINDING_COMPONENT_HPP_
#define _SCENE_MATERIAL_BINDING_COMPONENT_HPP_

#include <cstdint>

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
class MaterialBindingComponent {
  public:
    MaterialBindingComponent(const MaterialBindingComponent&) = delete;
    MaterialBindingComponent&
    operator=(const MaterialBindingComponent&) = delete;

    bool isAttached() const { return _owner != nullptr; }
    Prim* owner() const { return _owner; }

    Material* material() const;
    void setMaterial(Material* material);
    void clearMaterial();

    uint64_t version() const { return _version; }

  private:
    friend class Prim;

    explicit MaterialBindingComponent(Prim* owner);
    void detach();
    void requireAttached() const;
    void markChanged();

    Prim* _owner = nullptr;
    Material* _material = nullptr;
    uint64_t _version = 1;
};

} // namespace Scene
} // namespace KE

#endif
