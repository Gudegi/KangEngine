#ifndef _SCENE_RESOURCE_COMPONENT_HPP_
#define _SCENE_RESOURCE_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"

#include <string>
#include <cstdint>

namespace KE {
namespace Scene {

enum class ResourceType {
    Unknown,
    Mesh,
    Material,
    Texture,
    Shader,
};

using ResourceHandle = uint32_t;
static constexpr ResourceHandle InvalidResourceHandle = ~0u;

// User/editor visible resource entry attached to a Resource prim.
//
// ResourceComponent is metadata only. It describes the manager entry mirrored
// at `/.Resources/...`; the actual CPU/GPU payload remains owned by
// SceneResourceManager or by the renderable components that cache it.
class ResourceComponent : public ComponentBase {
  public:
    ResourceType type() const;
    void setType(ResourceType type);

    ResourceHandle handle() const;
    void setHandle(ResourceHandle handle);

    const std::string& uri() const;
    void setUri(std::string uri);

    const std::string& displayName() const;
    void setDisplayName(std::string name);

  private:
    friend class Prim;

    explicit ResourceComponent(Prim* owner);
    void detach();

    ResourceType _type = ResourceType::Unknown;
    ResourceHandle _handle = InvalidResourceHandle;
    std::string _uri;
    std::string _displayName;
};

const char* resourceTypeLabel(ResourceType type);

} // namespace Scene
} // namespace KE

#endif
