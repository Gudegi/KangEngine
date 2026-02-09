#ifndef _MJCF_LOADER_HPP_
#define _MJCF_LOADER_HPP_

#include "skeleton_tree.hpp"

#include <Eigen/Core>
#include <string>
#include <vector>

namespace KE {
namespace Animation {

struct MJCFMeshInfo {
    std::string bodyName;   // Body this mesh belongs to
    std::string meshFile;   // STL filename (e.g., "pelvis.stl")
    int bodyIndex;          // Index into SkeletonTree
};

struct MJCFLoadResult {
    SkeletonTree skeleton;
    std::vector<MJCFMeshInfo> meshInfos; // Visual mesh per body
    std::string meshDir;                  // Resolved mesh directory path
};

class MJCFLoader {
  public:
    // Load MJCF file and extract skeleton + mesh info
    static MJCFLoadResult load(const std::string& mjcfPath);
};

} // namespace Animation
} // namespace KE

#endif
