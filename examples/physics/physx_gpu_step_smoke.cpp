#include "physics/physics.hpp"
#include "physics/physics_gpu_system.hpp"

#include <cmath>
#include <fmt/base.h>
#include <PxActor.h>
#include <glm/gtc/quaternion.hpp>
#include <stdexcept>
#include <vector>

#ifdef KANGENGINE_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace KE;

namespace {

glm::quat zQuat(float angle) {
    return glm::angleAxis(angle, glm::vec3(0.0f, 0.0f, 1.0f));
}

float maxAbs(float a, float b) { return std::fabs(a - b); }

} // namespace

int main() {
    PhysicsConfig config = PhysicsConfig::zUp();
    config.enableGPU = true;
    config.gravity[0] = 0.0f;
    config.gravity[1] = 0.0f;
    config.gravity[2] = 0.0f;
    config.enableContactReports = false;

    PhysicsWorld world(config);
    for (int i = 0; i < 8; ++i) {
        const float x = static_cast<float>(i - 4) * 1.25f;
        auto* actor = world.createDynamicBox(
            glm::vec3(0.18f, 0.16f, 0.14f),
            glm::vec3(x, 0.35f * static_cast<float>(i % 3),
                      2.0f + 0.15f * static_cast<float>(i)),
            zQuat(0.07f * static_cast<float>(i)), 1.0f);
        actor->setLinearVelocity(
            PxVec3(0.04f * static_cast<float>(i + 1),
                   -0.03f * static_cast<float>(i % 2), 0.02f));
        actor->setAngularVelocity(PxVec3(0.0f));
        actor->wakeUp();
    }

    GpuPhysicsConfig gpuConfig;
    gpuConfig.cudaDeviceId = 0;
    PhysicsGpuSystem gpuSystem(&world, gpuConfig);
    gpuSystem.init();

    gpuSystem.fetchRigidData();

    const auto& view = gpuSystem.rigidData();
    fmt::print("C++ PhysX GPU smoke\n");
    fmt::print("  rigid shape : [{}, {}]\n", view.shape.at(0), view.shape.at(1));
    fmt::print("  version     : {}\n", view.version);
    fmt::print("  ptr         : {}\n", view.data);

#ifdef KANGENGINE_USE_CUDA
    std::vector<float> host(static_cast<size_t>(view.shape.at(0)) *
                            static_cast<size_t>(view.shape.at(1)));
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("cudaDeviceSynchronize failed");
    if (cudaMemcpy(host.data(), view.data, sizeof(float) * host.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy rigid data failed");

    std::vector<PxActor*> actors(static_cast<size_t>(view.shape.at(0)));
    const PxU32 actorCount = world.getScene()->getActors(
        PxActorTypeFlag::eRIGID_DYNAMIC, actors.data(),
        static_cast<PxU32>(actors.size()));

    float maxPositionError = 0.0f;
    float maxQuatError = 0.0f;
    for (PxU32 i = 0; i < actorCount; ++i) {
        auto* body = actors[i] ? actors[i]->is<PxRigidBody>() : nullptr;
        if (!body)
            throw std::runtime_error("expected rigid body actor");

        const PxTransform pose = body->getGlobalPose();
        PxVec3 expectedPosition = pose.p;
#ifdef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
        expectedPosition += body->getLinearVelocity() * config.dt;
#endif
        const size_t row = static_cast<size_t>(i) * 13;
        maxPositionError = std::max(maxPositionError,
                                    maxAbs(host[row + 0], expectedPosition.x));
        maxPositionError = std::max(maxPositionError,
                                    maxAbs(host[row + 1], expectedPosition.y));
        maxPositionError = std::max(maxPositionError,
                                    maxAbs(host[row + 2], expectedPosition.z));
        maxQuatError = std::max(maxQuatError, maxAbs(host[row + 3], pose.q.x));
        maxQuatError = std::max(maxQuatError, maxAbs(host[row + 4], pose.q.y));
        maxQuatError = std::max(maxQuatError, maxAbs(host[row + 5], pose.q.z));
        maxQuatError = std::max(maxQuatError, maxAbs(host[row + 6], pose.q.w));
    }
    fmt::print("  max pos err : {}\n", maxPositionError);
    fmt::print("  max quat err: {}\n", maxQuatError);
    if (maxPositionError > 1e-4f || maxQuatError > 1e-4f) {
        fmt::print("  first gpu row: [{}, {}, {}, {}, {}, {}, {}]\n",
                   host[0], host[1], host[2], host[3], host[4], host[5], host[6]);
        throw std::runtime_error("GPU rigid data does not match CPU actor state");
    }
#endif

    fmt::print("PASS: C++ PhysX GPU rigid fetch completed\n");
    return 0;
}
