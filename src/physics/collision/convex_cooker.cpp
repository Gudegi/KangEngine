#include "physics/collision/convex_cooker.hpp"

#include <cmath>
#include <cooking/PxCooking.h>
#include <stdexcept>
#include <string>

namespace KE {
namespace Physics {

std::vector<physx::PxConvexMesh*>
ConvexCollisionCooker::cook(physx::PxPhysics& physics,
                            const std::vector<ConvexMeshPart>& parts,
                            const ConvexCookingOptions& options) {
    using namespace physx;

    if (parts.empty())
        throw std::invalid_argument(
            "create convex collision requires at least one part");
    if (options.vertexLimit < 8 || options.vertexLimit > 255)
        throw std::invalid_argument("convex vertex_limit must be in [8, 255]");
    if (options.gpuCompatible && options.vertexLimit > 64)
        throw std::invalid_argument(
            "GPU-compatible convex vertex_limit must be <= 64");

    PxCookingParams params(physics.getTolerancesScale());
    params.buildGPUData = options.gpuCompatible;

    std::vector<PxConvexMesh*> meshes;
    meshes.reserve(parts.size());
    try {
        for (std::size_t partIndex = 0; partIndex < parts.size(); ++partIndex) {
            const auto& part = parts[partIndex];
            if (part.vertices.size() < 4) {
                throw std::invalid_argument("convex part " +
                                            std::to_string(partIndex) +
                                            " requires at least four vertices");
            }
            if (!part.indices.empty()) {
                if (part.indices.size() % 3 != 0)
                    throw std::invalid_argument(
                        "convex part " + std::to_string(partIndex) +
                        " indices must contain triangles");
                for (uint32_t index : part.indices) {
                    if (index >= part.vertices.size())
                        throw std::invalid_argument(
                            "convex part " + std::to_string(partIndex) +
                            " contains an out-of-range index");
                }
            }

            std::vector<PxVec3> points;
            points.reserve(part.vertices.size());
            for (const glm::vec3& vertex : part.vertices) {
                if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) ||
                    !std::isfinite(vertex.z)) {
                    throw std::invalid_argument(
                        "convex part " + std::to_string(partIndex) +
                        " contains a non-finite vertex");
                }
                points.emplace_back(vertex.x, vertex.y, vertex.z);
            }

            PxConvexMeshDesc desc;
            desc.points.data = points.data();
            desc.points.count = static_cast<PxU32>(points.size());
            desc.points.stride = sizeof(PxVec3);
            desc.vertexLimit = static_cast<PxU16>(options.vertexLimit);
            desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
            if (options.gpuCompatible)
                desc.flags |= PxConvexFlag::eGPU_COMPATIBLE;

            PxConvexMesh* mesh = PxCreateConvexMesh(
                params, desc, physics.getPhysicsInsertionCallback());
            if (!mesh) {
                throw std::runtime_error("PhysX failed to cook convex part " +
                                         std::to_string(partIndex));
            }
            meshes.push_back(mesh);
        }
    } catch (...) {
        for (PxConvexMesh* mesh : meshes)
            mesh->release();
        throw;
    }
    return meshes;
}

} // namespace Physics
} // namespace KE
