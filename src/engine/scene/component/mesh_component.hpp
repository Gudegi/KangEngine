#ifndef _SCENE_MESH_COMPONENT_HPP_
#define _SCENE_MESH_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"
#include "engine/scene/component/resource_component.hpp"

#include <memory>
#include <string>

namespace KE {
namespace Scene {

struct MeshData;

// Mesh resource binding attached to a renderable Prim.
//
// This component is the migration point from "Prim directly owns mesh data" to
// "Prim owns components that describe renderable state".  For now it supports
// the direct MeshData path used by renderer registration and the
// mesh-source-path instancing path.
//
// Ownership model:
// - SceneResourceManager is the conceptual owner/catalog for shared mesh
//   resources: stable handle, URI/name metadata, Resource Prim mirror, and the
//   shared MeshData retention entry.
// - MeshComponent is the per-Prim binding to one mesh resource.  It may keep a
//   resolved MeshData shared_ptr so the render path does not need to look up
//   the manager every frame, but that cache is not the resource identity
//   source.
// - Resource Prim is editor-visible metadata only; it never owns geometry and
// is
//   not renderable/manipulatable.
class MeshComponent : public ComponentBase {
  public:
    void setMeshData(std::shared_ptr<MeshData> data);
    std::shared_ptr<MeshData> meshData() const;

    void setMeshSourcePath(std::string path);
    const std::string& meshSourcePath() const;

    void setResourceHandle(ResourceHandle handle);
    ResourceHandle resourceHandle() const;

    std::shared_ptr<MeshData> resolveMeshData() const;

  private:
    friend class Prim;

    explicit MeshComponent(Prim* owner);
    void detach();
    void markGeometryChanged();

    // Fast cached mesh pointer for renderer registration.  The matching
    // ResourceHandle remains the identity used for editor/resource management.
    std::shared_ptr<MeshData> _meshData;
    std::string _meshSourcePath;
    ResourceHandle _resourceHandle = InvalidResourceHandle;
    mutable std::weak_ptr<MeshData> _resolvedMeshDataCache;
};

} // namespace Scene
} // namespace KE

#endif
