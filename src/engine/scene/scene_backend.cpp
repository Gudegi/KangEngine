///
/// Scene Backend Factory Implementation
///

#include "scene_backend.hpp"
#include "animation/skeleton_math.hpp"
#include "animation/skeleton_tree.hpp"
#include "native/native_scene.hpp"
#include "native/prim.hpp"

#ifdef KANGENGINE_USE_USD
#include "usd/usd_scene.hpp"
#endif

#include <stdexcept>

namespace KE {
namespace Scene {

std::unique_ptr<SceneBackend> SceneFactory::createBackend(BackendType type) {
    switch (type) {
        case BackendType::Native:
            return std::make_unique<NativeScene>();

        case BackendType::USD:
        #ifdef KANGENGINE_USE_USD
            return std::make_unique<USDScene>();
        #else
            throw std::runtime_error("USD support not compiled. Rebuild with -DUSE_USD=ON");
        #endif

        default:
            throw std::runtime_error("Unknown scene backend type");
    }
}

std::vector<Prim*> defineSkeletonTree(SceneBackend* scene,
                                      const std::string& basePath,
                                      const Animation::SkeletonTree& tree) {
    if (!scene)
        throw std::runtime_error("defineSkeletonTree requires scene");

    const int jointCount = tree.numJoints();
    std::vector<Prim*> jointPrims(static_cast<size_t>(jointCount), nullptr);
    const std::vector<std::string> jointPaths = tree.nodePaths(basePath);

    scene->definePrim(basePath, PrimType::Xform);

    for (int i = 0; i < jointCount; ++i) {
        Prim* prim = scene->definePrim(jointPaths[static_cast<size_t>(i)],
                                       PrimType::Xform);
        prim->setLocalTranslation(Animation::toGlm(tree.localTranslation(i)));
        prim->setLocalRotation(Animation::toGlm(tree.localRotation(i)));
        jointPrims[static_cast<size_t>(i)] = prim;
    }

    return jointPrims;
}

} // namespace Scene
} // namespace KE
