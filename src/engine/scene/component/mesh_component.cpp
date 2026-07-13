#include "mesh_component.hpp"

#include "engine/scene/component/render_component.hpp"
#include "engine/scene/component/resource_component.hpp"
#include "engine/scene/native/prim.hpp"

#include <utility>

namespace KE {
namespace Scene {

MeshComponent::MeshComponent(Prim* owner)
    : ComponentBase(owner, "MeshComponent") {}

void MeshComponent::detach() {
    if (!_owner)
        return;
    detachBase();
}

void MeshComponent::markGeometryChanged() {
    _resolvedMeshDataCache.reset();
    markChanged();
    if (_owner) {
        if (auto render = _owner->getRenderComponent())
            render->markChanged();
    }
}

void MeshComponent::setMeshData(std::shared_ptr<MeshData> data) {
    requireAttached();
    if (_meshData == data)
        return;
    _meshData = std::move(data);
    markGeometryChanged();
}

std::shared_ptr<MeshData> MeshComponent::meshData() const {
    requireAttached();
    return _meshData;
}

void MeshComponent::setMeshSourcePath(std::string path) {
    requireAttached();
    if (_meshSourcePath == path)
        return;
    _meshSourcePath = std::move(path);
    markGeometryChanged();
}

const std::string& MeshComponent::meshSourcePath() const {
    requireAttached();
    return _meshSourcePath;
}

void MeshComponent::setResourceHandle(ResourceHandle handle) {
    requireAttached();
    if (_resourceHandle == handle)
        return;
    _resourceHandle = handle;
    markGeometryChanged();
}

ResourceHandle MeshComponent::resourceHandle() const {
    requireAttached();
    return _resourceHandle;
}

std::shared_ptr<MeshData> MeshComponent::resolveMeshData() const {
    requireAttached();
    if (_meshData)
        return _meshData;

    if (_owner->getType() != PrimType::MeshInstance || _meshSourcePath.empty())
        return nullptr;

    if (auto cached = _resolvedMeshDataCache.lock())
        return cached;

    const Prim* root = _owner;
    while (root->getParent())
        root = root->getParent();

    auto* source = const_cast<Prim*>(root)->getPrimAtPath(_meshSourcePath);
    if (!source || source == _owner)
        return nullptr;

    auto resolved = source->resolveMeshData();
    _resolvedMeshDataCache = resolved;
    return resolved;
}

} // namespace Scene
} // namespace KE
