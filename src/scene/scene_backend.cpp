///
/// Scene Backend Factory Implementation
///

#include "scene_backend.hpp"
#include "native/native_scene.hpp"

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

} // namespace Scene
} // namespace KE
