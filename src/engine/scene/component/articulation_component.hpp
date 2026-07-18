#ifndef _SCENE_ARTICULATION_COMPONENT_HPP_
#define _SCENE_ARTICULATION_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"

#include <string>

namespace KE {
namespace Scene {

// Root-level metadata for an articulated object subtree.
//
// This component intentionally does not own PhysX handles, FK state, or body
// prim pointers. Runtime ownership currently stays in KangSimWorld /
// SkeletonBridge / PhysicsBridge. The component marks the root Prim as the
// stable scene identity for that articulation and stores lightweight import /
// instancing metadata that inspectors and future systems can use.
class ArticulationComponent : public ComponentBase {
  public:
    const std::string& assetPath() const { return _assetPath; }
    void setAssetPath(std::string assetPath);

    const std::string& rootPath() const { return _rootPath; }
    void setRootPath(std::string rootPath);

    const std::string& meshAssetBasePath() const { return _meshAssetBasePath; }
    void setMeshAssetBasePath(std::string meshAssetBasePath);

    int bodyCount() const { return _bodyCount; }
    void setBodyCount(int bodyCount);

    int renderPrimCount() const { return _renderPrimCount; }
    void setRenderPrimCount(int renderPrimCount);

    bool splitVisualGeoms() const { return _splitVisualGeoms; }
    void setSplitVisualGeoms(bool splitVisualGeoms);

    void setArticulationMetadata(std::string rootPath, std::string assetPath,
                                 std::string meshAssetBasePath, int bodyCount,
                                 int renderPrimCount, bool splitVisualGeoms);

  private:
    friend class Prim;

    explicit ArticulationComponent(Prim* owner);
    void detach();

    std::string _assetPath;
    std::string _rootPath;
    std::string _meshAssetBasePath;
    int _bodyCount = 0;
    int _renderPrimCount = 0;
    bool _splitVisualGeoms = false;
};

} // namespace Scene
} // namespace KE

#endif
