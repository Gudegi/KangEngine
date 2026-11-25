#ifndef _SCENE_MESH_COMPONENT_HPP_
#define _SCENE_MESH_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"
#include "engine/scene/component/resource_component.hpp"

#include <memory>
#include <string>

namespace KE {
namespace Scene {

struct MeshData;

// Binds a Prim to source MeshData or a path to another mesh Prim.
//
// Ownership:
// - SceneResourceManager assigns resource handles and retains registered meshes.
// - This component retains direct MeshData and weakly caches meshes resolved
//   from a source Prim path.
// - Cached pointers do not replace resource identity. The backend owns GPU meshes.
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

    // Retains source geometry; runtime deformation must not modify it.
    std::shared_ptr<MeshData> _meshData;
    std::string _meshSourcePath;
    ResourceHandle _resourceHandle = InvalidResourceHandle;
    mutable std::weak_ptr<MeshData> _resolvedMeshDataCache;
};

} // namespace Scene
} // namespace KE

#endif
