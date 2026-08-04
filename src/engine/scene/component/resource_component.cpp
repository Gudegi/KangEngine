#include "resource_component.hpp"

#include <utility>

namespace KE {
namespace Scene {

ResourceComponent::ResourceComponent(Prim* owner)
    : ComponentBase(owner, "ResourceComponent") {}

void ResourceComponent::detach() {
    if (!_owner)
        return;
    detachBase();
}

ResourceType ResourceComponent::type() const {
    requireAttached();
    return _type;
}

void ResourceComponent::setType(ResourceType type) {
    requireAttached();
    if (_type == type)
        return;
    _type = type;
    markChanged();
}

ResourceHandle ResourceComponent::handle() const {
    requireAttached();
    return _handle;
}

void ResourceComponent::setHandle(ResourceHandle handle) {
    requireAttached();
    if (_handle == handle)
        return;
    _handle = handle;
    markChanged();
}

const std::string& ResourceComponent::uri() const {
    requireAttached();
    return _uri;
}

void ResourceComponent::setUri(std::string uri) {
    requireAttached();
    if (_uri == uri)
        return;
    _uri = std::move(uri);
    markChanged();
}

const std::string& ResourceComponent::displayName() const {
    requireAttached();
    return _displayName;
}

void ResourceComponent::setDisplayName(std::string name) {
    requireAttached();
    if (_displayName == name)
        return;
    _displayName = std::move(name);
    markChanged();
}

const char* resourceTypeLabel(ResourceType type) {
    switch (type) {
    case ResourceType::Unknown:
        return "Unknown";
    case ResourceType::Mesh:
        return "Mesh";
    case ResourceType::Material:
        return "Material";
    case ResourceType::Texture:
        return "Texture";
    case ResourceType::ShaderSource:
        return "Shader Source";
    case ResourceType::Pipeline:
        return "Pipeline";
    }
    return "Unknown";
}

} // namespace Scene
} // namespace KE
