#include "physics/physics.hpp"
#include "physics/physics_gpu_system.hpp"
#include "asset/articulation_desc.hpp"
#include "asset/heightmap_loader.hpp"
#include "engine/scene/scene_backend.hpp"
#ifdef KANGENGINE_USE_CUDA
#include <cuda_runtime.h>
#endif
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace KE;

// Each case runs in a fresh process: a failed CUDA context cannot be reused.
// terrain_collision_smoke [cpu|gpu] [plane|flat|rough|mesh_flat|mesh_rough]
//                         [bodies] [samples] [steps] [box|sphere] [interior|edge]
int main(int argc, char** argv) {
    const bool gpu = argc > 1 && std::string(argv[1]) == "gpu";
    const std::string terrain = argc > 2 ? argv[2] : "rough";
    const int bodies = argc > 3 ? std::stoi(argv[3]) : 256;
    const int samples = argc > 4 ? std::stoi(argv[4]) : 257;
    const int steps = argc > 5 ? std::stoi(argv[5]) : 600;
    if (bodies < 1 || samples < 2 || steps < 1)
        throw std::invalid_argument("positive bodies/steps and samples >= 2 required");
    PhysicsConfig config = PhysicsConfig::zUp();
    config.enableGPU = gpu;
    config.enableContactReports = false;
    config.dt = 0.005f;
    PhysicsWorld world(config);
    if (gpu && !(world.getScene()->getFlags() & PxSceneFlag::eENABLE_GPU_DYNAMICS))
        throw std::runtime_error("GPU requested but unavailable");
    constexpr float extent = 160.f;
    if (terrain == "plane") {
        world.addDefaultGround();
    } else {
        std::vector<float> heights(size_t(samples) * samples);
        for (int r = 0; r < samples; ++r)
            for (int c = 0; c < samples; ++c)
                heights[size_t(r) * samples + c] = (terrain == "flat" || terrain == "mesh_flat") ? 0.f :
                    0.08f * std::sin(float(r) * 0.37f) * std::cos(float(c) * 0.29f);
        if (terrain.rfind("mesh_", 0) == 0) {
            Asset::HeightFieldMeshOptions options;
            options.upAxis = UpAxis::Z;
            options.horizontalScale = extent / (samples - 1);
            auto mesh = std::make_shared<Scene::MeshData>(
                Asset::heightFieldToMesh(heights.data(), samples, samples, options));
            world.createStaticTriangleMesh(mesh, glm::vec3(0), glm::quat(1, 0, 0, 0),
                                            Physics::PhysicsMaterialDesc{});
        } else if (!world.createStaticHeightField(heights.data(), samples, samples,
                extent / (samples - 1), Physics::PhysicsMaterialDesc{}, UpAxis::Z))
            throw std::runtime_error("heightfield creation failed");
    }
    const int columns = int(std::ceil(std::sqrt(float(bodies))));
    for (int i = 0; i < bodies; ++i) {
        const float x = argc > 7 && std::string(argv[7]) == "edge" ? -80.05f :
            (float(i % columns) + 0.5f) * 140.f / columns - 70.f;
        const float y = (float(i / columns) + 0.5f) * 140.f / columns - 70.f;
        if (argc > 6 && std::string(argv[6]) == "sphere")
            world.createDynamicSphere(0.15f, glm::vec3(x, y, 0.5f));
        else
            world.createDynamicBox(glm::vec3(0.15f), glm::vec3(x, y, 0.5f));
    }
    std::cout << "case gpu=" << gpu << " terrain=" << terrain << " bodies="
              << bodies << " samples=" << samples << std::endl;
    for (int step = 0; step < steps; ++step) {
        world.getScene()->simulate(config.dt);
        PxU32 error = 0;
        if (!world.getScene()->fetchResults(true, &error) || error)
            throw std::runtime_error("fetchResults failed at step " + std::to_string(step));
#ifdef KANGENGINE_USE_CUDA
        if (gpu && cudaDeviceSynchronize() != cudaSuccess)
            throw std::runtime_error("CUDA error at step " + std::to_string(step));
#endif
        if (step % 100 == 0 || step + 1 == steps) {
            PxSimulationStatistics stats;
            world.getScene()->getSimulationStatistics(stats);
            std::cout << "step=" << step << " contacts=" << stats.nbDiscreteContactPairsTotal;
#if PX_PHYSICS_VERSION_MAJOR == 5 && PX_PHYSICS_VERSION_MINOR >= 8
            const auto& memory = stats.gpuDynamicsMemoryConfigStatistics;
            std::cout << " rigid_contacts=" << memory.rigidContactCount
                      << " patches=" << memory.rigidPatchCount
                      << " collision_stack=" << memory.collisionStackSize
                      << " heap=" << stats.gpuMemHeap;
#endif
            std::cout << std::endl;
        }
    }
#ifdef KANGENGINE_USE_CUDA
    if (gpu) {
        PhysicsGpuSystem state(&world, GpuPhysicsConfig{});
        state.init();
        state.fetchRigidData();
        const auto& view = state.rigidData();
        std::vector<float> host(size_t(bodies) * 13);
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(host.data(), view.data, host.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("rigid state readback failed");
        float minHeight = host[2];
        for (int i = 0; i < bodies; ++i)
            minHeight = std::min(minHeight, host[size_t(i) * 13 + 2]);
        std::cout << "minimum_body_height=" << minHeight << std::endl;
        if (!(argc > 7 && std::string(argv[7]) == "edge") && minHeight < -0.1f)
            throw std::runtime_error("a body fell through the terrain");
    }
#endif
    std::cout << "PASS" << std::endl;
}
