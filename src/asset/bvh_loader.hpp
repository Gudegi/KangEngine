#ifndef _BVH_LOADER_HPP_
#define _BVH_LOADER_HPP_

#include "animation/skeleton_motion.hpp"
#include "animation/skeleton_tree.hpp"
#include "asset/import_diagnostics.hpp"

#include <string>

namespace KE {
namespace Asset {

struct BVHImportResult {
    Animation::SkeletonMotion motion;
    ImportDiagnostics diagnostics;
    int frameCount = 0;
    float frameTime = 0.0f;
    float frameRate = 0.0f;
};

class BVHLoader {
  public:
    BVHLoader() = delete;

    static Animation::SkeletonTree loadSkeleton(const std::string& bvhPath,
                                                float scale = 1.0f);

    static Animation::SkeletonMotion loadMotion(const std::string& bvhPath,
                                                float scale = 1.0f);

    static BVHImportResult parse(const std::string& bvhPath,
                                 float scale = 1.0f);
};

} // namespace Asset
} // namespace KE

#endif
