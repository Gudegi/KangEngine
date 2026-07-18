#include "engine/scene/component/articulation_component.hpp"

#include <utility>

namespace KE {
namespace Scene {

ArticulationComponent::ArticulationComponent(Prim* owner)
    : ComponentBase(owner, "ArticulationComponent") {}

void ArticulationComponent::detach() { detachBase(); }

void ArticulationComponent::setAssetPath(std::string assetPath) {
    requireAttached();
    if (_assetPath == assetPath)
        return;
    _assetPath = std::move(assetPath);
    markChanged();
}

void ArticulationComponent::setRootPath(std::string rootPath) {
    requireAttached();
    if (_rootPath == rootPath)
        return;
    _rootPath = std::move(rootPath);
    markChanged();
}

void ArticulationComponent::setMeshAssetBasePath(
    std::string meshAssetBasePath) {
    requireAttached();
    if (_meshAssetBasePath == meshAssetBasePath)
        return;
    _meshAssetBasePath = std::move(meshAssetBasePath);
    markChanged();
}

void ArticulationComponent::setBodyCount(int bodyCount) {
    requireAttached();
    if (_bodyCount == bodyCount)
        return;
    _bodyCount = bodyCount;
    markChanged();
}

void ArticulationComponent::setRenderPrimCount(int renderPrimCount) {
    requireAttached();
    if (_renderPrimCount == renderPrimCount)
        return;
    _renderPrimCount = renderPrimCount;
    markChanged();
}

void ArticulationComponent::setSplitVisualGeoms(bool splitVisualGeoms) {
    requireAttached();
    if (_splitVisualGeoms == splitVisualGeoms)
        return;
    _splitVisualGeoms = splitVisualGeoms;
    markChanged();
}

void ArticulationComponent::setArticulationMetadata(
    std::string rootPath, std::string assetPath,
    std::string meshAssetBasePath, int bodyCount, int renderPrimCount,
    bool splitVisualGeoms) {
    requireAttached();
    const bool changed = _rootPath != rootPath || _assetPath != assetPath ||
                         _meshAssetBasePath != meshAssetBasePath ||
                         _bodyCount != bodyCount ||
                         _renderPrimCount != renderPrimCount ||
                         _splitVisualGeoms != splitVisualGeoms;
    if (!changed)
        return;
    _rootPath = std::move(rootPath);
    _assetPath = std::move(assetPath);
    _meshAssetBasePath = std::move(meshAssetBasePath);
    _bodyCount = bodyCount;
    _renderPrimCount = renderPrimCount;
    _splitVisualGeoms = splitVisualGeoms;
    markChanged();
}

} // namespace Scene
} // namespace KE
