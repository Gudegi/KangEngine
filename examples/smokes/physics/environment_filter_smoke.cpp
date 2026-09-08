#include "asset/mjcf_loader.hpp"
#include "physics/articulation.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace KE;

#if PX_PHYSICS_VERSION_MAJOR > 5 || (PX_PHYSICS_VERSION_MAJOR == 5 && PX_PHYSICS_VERSION_MINOR >= 8)
static PxFilterFlags customFilter(PxFilterObjectAttributes a, PxFilterData fa,
                                 PxFilterObjectAttributes b, PxFilterData fb,
                                 PxPairFlags& pair, const void* block, PxU32 size) {
    return PxDefaultSimulationFilterShader(a, fa, b, fb, pair, block, size);
}

int main(int argc, char** argv) {
    if (argc < 3)
        throw std::invalid_argument("environment_filter_smoke cpu|gpu MJCF [custom]");
    const bool gpu = std::string(argv[1]) == "gpu";
    const bool custom = argc > 3;
    PhysicsConfig config = PhysicsConfig::zUp();
    config.enableGPU = gpu;
    if (custom)
        config.filterShader = customFilter;
    PhysicsWorld world(config);
    if (gpu && world.getScene()->getBroadPhaseType() != PxBroadPhaseType::eGPU)
        throw std::runtime_error("GPU broadphase unavailable");
    world.addDefaultGround();
    const auto data = Asset::MJCFLoader::load(argv[2]);
    for (const bool aggregate : {false, true}) {
        for (const PxU32 group : {0u, 1u, 2u, (1u << 24) - 1, 1u << 24, (1u << 24) + 1}) {
            auto cfg = ArticulationConfig::freeBase();
            cfg.useAggregate = aggregate;
            cfg.collisionGroup = group;
            auto articulation = Articulation::build(
                world, data.skeletonTree, data.collisionGeoms, data.joints,
                data.inertials, cfg);
            const PxU32 expected = gpu && !custom && group > 0 && group < (1u << 24)
                                       ? group - 1 : PX_INVALID_U32;
            for (auto* link : articulation.links()) {
                std::vector<PxShape*> shapes(link->getNbShapes());
                link->getShapes(shapes.data(), static_cast<PxU32>(shapes.size()));
                for (auto* shape : shapes) {
                    if (gpu && shape->getGeometry().getType() == PxGeometryType::eCONVEXMESH) {
                        const auto& convex = static_cast<const PxConvexMeshGeometry&>(shape->getGeometry());
                        if (!convex.convexMesh->isGpuCompatible())
                            throw std::runtime_error("Imported convex collision is not GPU compatible");
                    }
                }
                const auto* owner = link->getAggregate();
                const PxU32 actual = owner ? owner->getEnvironmentID()
                                           : link->getEnvironmentID();
                if (actual != expected)
                    throw std::runtime_error("Incorrect broadphase environment ID: aggregate=" + std::to_string(aggregate) + " group=" + std::to_string(group) + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
                if (owner && link->getEnvironmentID() != PX_INVALID_U32)
                    throw std::runtime_error("Aggregate members must retain wildcard IDs");
            }
        }
    }
    std::cout << "PASS: aggregate/link IDs, shared group, range fallback, "
              << (custom ? "custom" : "built-in") << " filter, "
              << (gpu ? "GPU" : "CPU") << std::endl;
}
#else
int main() { std::cout << "SKIP: environment filtering requires PhysX 5.8+" << std::endl; }
#endif
