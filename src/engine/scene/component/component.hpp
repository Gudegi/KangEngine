#ifndef _SCENE_COMPONENT_HPP_
#define _SCENE_COMPONENT_HPP_

#include <cstdint>
#include <stdexcept>
#include <string>

namespace KE {
namespace Scene {

class Prim;

// Lightweight base for explicit Prim-owned components.
//
// This is intentionally not a full ECS abstraction: there is no global
// registry, runtime type table, or virtual update loop here. The base only
// captures the lifecycle contract that all current Prim components already
// share:
//
// - non-owning Prim owner pointer
// - attached/detached state
// - monotonically increasing version for dirty/change tracking
// - consistent detached-component errors
class ComponentBase {
  public:
    ComponentBase(const ComponentBase&) = delete;
    ComponentBase& operator=(const ComponentBase&) = delete;
    ComponentBase(ComponentBase&&) = delete;
    ComponentBase& operator=(ComponentBase&&) = delete;

    bool isAttached() const { return _owner != nullptr; }
    Prim* owner() const { return _owner; }
    uint64_t version() const { return _version; }

  protected:
    ComponentBase(Prim* owner, const char* componentName)
        : _owner(owner), _componentName(componentName) {
        requireAttached();
    }

    ~ComponentBase() = default;

    void requireAttached() const {
        if (!_owner)
            throw std::runtime_error(std::string(_componentName) +
                                     " is detached from its Prim");
    }

    void markChanged() { ++_version; }

    void detachBase() {
        if (!_owner)
            return;
        _owner = nullptr;
        markChanged();
    }

    Prim* _owner = nullptr;

  private:
    const char* _componentName = "Component";
    uint64_t _version = 1;
};

} // namespace Scene
} // namespace KE

#endif
