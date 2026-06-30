#include "physics/physics.hpp"
#include "physics/physics_gpu_system.hpp"
#include "sim/gpu_transform_kernels.hpp"

#include <cmath>
#include <fmt/base.h>
#include <PxActor.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
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

float absDiff(float a, float b) { return std::fabs(a - b); }

} // namespace

int main() {
    PhysicsConfig config = PhysicsConfig::zUp();
    config.enableGPU = true;
    config.gravity[0] = 0.0f;
    config.gravity[1] = 0.0f;
    config.gravity[2] = -9.8f;
    config.enableContactReports = false;

    PhysicsWorld world(config);
    for (int i = 0; i < 8; ++i) {
        const float x = static_cast<float>(i - 4) * 1.25f;
        auto* actor = world.createDynamicBox(
            glm::vec3(0.18f, 0.16f, 0.14f),
            glm::vec3(x, 0.35f * static_cast<float>(i % 3),
                      2.0f + 0.15f * static_cast<float>(i)),
            zQuat(0.07f * static_cast<float>(i)), 1.0f);
        actor->setLinearVelocity(PxVec3(0.04f * static_cast<float>(i + 1),
                                        -0.03f * static_cast<float>(i % 2),
                                        0.02f));
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
    fmt::print("  rigid shape : [{}, {}]\n", view.shape.at(0),
               view.shape.at(1));
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

    const PxU32 actorCount = static_cast<PxU32>(view.shape.at(0));
    if (actorCount != 8)
        throw std::runtime_error("unexpected rigid actor count");

    float maxPositionError = 0.0f;
    float maxQuatError = 0.0f;
    float maxVelocityError = 0.0f;
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        const PxVec3 initialPosition(
            static_cast<float>(static_cast<int>(i) - 4) * 1.25f,
            0.35f * static_cast<float>(i % 3),
            2.0f + 0.15f * static_cast<float>(i));
        const PxVec3 initialVelocity(0.04f * static_cast<float>(i + 1),
                                     -0.03f * static_cast<float>(i % 2), 0.02f);
        const PxVec3 expectedVelocity(
            initialVelocity.x + config.gravity[0] * config.dt,
            initialVelocity.y + config.gravity[1] * config.dt,
            initialVelocity.z + config.gravity[2] * config.dt);
        const PxVec3 expectedPosition =
            initialPosition + expectedVelocity * config.dt;
        const glm::quat expectedRotation = zQuat(0.07f * static_cast<float>(i));
        maxPositionError = std::max(maxPositionError,
                                    absDiff(host[row + 0], expectedPosition.x));
        maxPositionError = std::max(maxPositionError,
                                    absDiff(host[row + 1], expectedPosition.y));
        maxPositionError = std::max(maxPositionError,
                                    absDiff(host[row + 2], expectedPosition.z));
        maxQuatError =
            std::max(maxQuatError, absDiff(host[row + 3], expectedRotation.x));
        maxQuatError =
            std::max(maxQuatError, absDiff(host[row + 4], expectedRotation.y));
        maxQuatError =
            std::max(maxQuatError, absDiff(host[row + 5], expectedRotation.z));
        maxQuatError =
            std::max(maxQuatError, absDiff(host[row + 6], expectedRotation.w));
        maxVelocityError = std::max(maxVelocityError,
                                    absDiff(host[row + 7], expectedVelocity.x));
        maxVelocityError = std::max(maxVelocityError,
                                    absDiff(host[row + 8], expectedVelocity.y));
        maxVelocityError = std::max(maxVelocityError,
                                    absDiff(host[row + 9], expectedVelocity.z));
    }
    Sim::CUDAExternalTransformBuffer transformBuffer(
        static_cast<int>(actorCount), gpuConfig.cudaDeviceId,
        "physx_gpu_step_smoke_transforms");
    Sim::launchRigidStateToMat4CUDA(view, transformBuffer.view(),
                                    static_cast<int>(actorCount));
    transformBuffer.incrementVersion();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after rigid state to mat4 failed");

    std::vector<float> transformHost(static_cast<size_t>(actorCount) * 16);
    if (cudaMemcpy(transformHost.data(), transformBuffer.view().data,
                   sizeof(float) * transformHost.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy transform data failed");

    float maxTransformError = 0.0f;
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        const glm::vec3 pos(host[row + 0], host[row + 1], host[row + 2]);
        const glm::quat rot(host[row + 6], host[row + 3], host[row + 4],
                            host[row + 5]);
        const glm::mat4 expected =
            glm::translate(glm::mat4(1.0f), pos) * glm::mat4_cast(rot);
        const float* expectedData = glm::value_ptr(expected);
        const size_t matrix = static_cast<size_t>(i) * 16;
        for (size_t component = 0; component < 16; ++component) {
            maxTransformError = std::max(
                maxTransformError,
                absDiff(transformHost[matrix + component],
                        expectedData[component]));
        }
    }
    fmt::print("  mat4 err    : {}\n", maxTransformError);
    if (maxTransformError > 1e-4f)
        throw std::runtime_error(
            "CUDA rigid state to Mat4 transform does not match CPU layout");

    fmt::print("  max pos err : {}\n", maxPositionError);
    fmt::print("  max quat err: {}\n", maxQuatError);
    fmt::print("  max vel err : {}\n", maxVelocityError);
    if (maxPositionError > 1e-4f || maxQuatError > 1e-4f ||
        maxVelocityError > 1e-4f) {
        fmt::print("  first gpu row: [{}, {}, {}, {}, {}, {}, {}]\n", host[0],
                   host[1], host[2], host[3], host[4], host[5], host[6]);
        throw std::runtime_error(
            "GPU rigid data does not match CPU actor state");
    }

    std::vector<float> applied(host.size(), 0.0f);
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        const glm::quat rotation = zQuat(0.11f * static_cast<float>(i));
        applied[row + 0] = -8.0f + 2.0f * static_cast<float>(i);
        applied[row + 1] = 1.0f + 0.2f * static_cast<float>(i);
        applied[row + 2] = 4.0f + 0.1f * static_cast<float>(i);
        applied[row + 3] = rotation.x;
        applied[row + 4] = rotation.y;
        applied[row + 5] = rotation.z;
        applied[row + 6] = rotation.w;
        applied[row + 7] = 0.1f * static_cast<float>(i + 1);
        applied[row + 8] = -0.04f * static_cast<float>(i);
        applied[row + 9] = 0.03f;
    }

    if (cudaMemcpy(view.data, applied.data(), sizeof(float) * applied.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy applied rigid data failed");
    gpuSystem.applyRigidData();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("cudaDeviceSynchronize after apply failed");

    world.step();
    gpuSystem.fetchRigidData();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("cudaDeviceSynchronize after fetch failed");
    if (cudaMemcpy(host.data(), view.data, sizeof(float) * host.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy applied rigid result failed");

    float maxAppliedPositionError = 0.0f;
    float maxAppliedQuatError = 0.0f;
    float maxAppliedVelocityError = 0.0f;
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        for (size_t axis = 0; axis < 3; ++axis) {
            const float expectedVelocity =
                applied[row + 7 + axis] + config.gravity[axis] * config.dt;
            const float expectedPosition =
                applied[row + axis] + expectedVelocity * config.dt;
            maxAppliedPositionError =
                std::max(maxAppliedPositionError,
                         absDiff(host[row + axis], expectedPosition));
            maxAppliedVelocityError =
                std::max(maxAppliedVelocityError,
                         absDiff(host[row + 7 + axis], expectedVelocity));
            maxAppliedVelocityError = std::max(
                maxAppliedVelocityError,
                absDiff(host[row + 10 + axis], applied[row + 10 + axis]));
        }
        for (size_t component = 3; component < 7; ++component)
            maxAppliedQuatError = std::max(
                maxAppliedQuatError,
                absDiff(host[row + component], applied[row + component]));
    }

    fmt::print("  apply pos err: {}\n", maxAppliedPositionError);
    fmt::print("  apply quat err: {}\n", maxAppliedQuatError);
    fmt::print("  apply vel err: {}\n", maxAppliedVelocityError);
    if (maxAppliedPositionError > 1e-4f || maxAppliedQuatError > 1e-4f ||
        maxAppliedVelocityError > 1e-4f)
        throw std::runtime_error("GPU rigid apply/step/fetch round trip did "
                                 "not match expected state");
#endif

    fmt::print("PASS: C++ PhysX GPU rigid fetch/apply round trip completed\n");
    return 0;
}
