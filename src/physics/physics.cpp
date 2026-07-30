#include "physics.hpp"
#include "PxBroadPhase.h"
#include "PxSceneDesc.h"
#include "character/character_description.hpp"
#include "articulation.hpp"
#include "collision_material_utils.hpp"
#include "physics/physx_compat.hpp"
#include <cooking/PxCooking.h>
#ifndef __APPLE__
#include "gpu/PxGpu.h"
#endif
#ifdef KANGENGINE_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#endif
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace KE {

namespace {

PxTransform rigidFromToPose(const Eigen::Vector3f& from,
                            const Eigen::Vector3f& to) {
    Eigen::Vector3f mid = (from + to) * 0.5f;
    Eigen::Vector3f dir = to - from;
    float len = dir.norm();
    if (len < 1e-6f)
        return PxTransform(PxVec3(mid.x(), mid.y(), mid.z()));
    Eigen::Quaternionf q =
        Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), dir / len);
    return PxTransform(PxVec3(mid.x(), mid.y(), mid.z()),
                       PxQuat(q.x(), q.y(), q.z(), q.w()));
}

PxQuat rigidMjcfShapeRot(const Eigen::Quaternionf& mjcfQuat) {
    Eigen::Vector3f axis = mjcfQuat * Eigen::Vector3f::UnitZ();
    Eigen::Quaternionf q =
        Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), axis);
    return PxQuat(q.x(), q.y(), q.z(), q.w());
}

void applyRigidContactOffsets(PxShape* shape, float contactOffset,
                              float restOffset) {
    if (!shape)
        return;
    shape->setRestOffset(restOffset);
    shape->setContactOffset(std::max(contactOffset, restOffset + 1e-4f));
}

void setRigidCollisionFilterData(PxRigidActor* actor, PxU32 collisionGroup) {
    if (!actor || collisionGroup == 0)
        return;

    PxFilterData filterData;
    filterData.word2 = collisionGroup;
    filterData.word3 = 1;

    const PxU32 numShapes = actor->getNbShapes();
    std::vector<PxShape*> shapes(numShapes);
    actor->getShapes(shapes.data(), numShapes);
    for (PxShape* shape : shapes) {
        if (!shape)
            continue;
        shape->setSimulationFilterData(filterData);
        shape->setQueryFilterData(filterData);
    }
}

PxTransform heightFieldPose(int rows, int cols, float horizontalScale,
                            UpAxis upAxis, bool center) {
    const float originX =
        center ? (static_cast<float>(cols - 1) * 0.5f * horizontalScale) : 0.f;
    const float originZ =
        center ? (static_cast<float>(rows - 1) * 0.5f * horizontalScale) : 0.f;

    if (upAxis == UpAxis::Z) {
        // PhysX heightfields are local Y-up. Rotate local Y to world Z, then
        // reverse the row/column mapping below so source rows still increase
        // along positive world Y like heightFieldToMesh(..., UpAxis::Z).
        return PxTransform(
            PxVec3(-originX,
                   (static_cast<float>(rows - 1) * horizontalScale) - originZ,
                   0.f),
            PxQuat(PxHalfPi, PxVec3(1.f, 0.f, 0.f)));
    }

    return PxTransform(PxVec3(-originX, 0.f, -originZ));
}

std::vector<PxHeightFieldSample> makeHeightFieldSamples(const float* heights,
                                                        int rows, int cols,
                                                        UpAxis upAxis,
                                                        float& outHeightScale) {
    float maxAbsHeight = 0.f;
    const int count = rows * cols;
    for (int i = 0; i < count; ++i)
        maxAbsHeight = std::max(maxAbsHeight, std::abs(heights[i]));

    outHeightScale = maxAbsHeight > 1e-6f
                         ? maxAbsHeight / static_cast<float>(
                                              std::numeric_limits<PxI16>::max())
                         : 1.0f;

    // PhysX local X is sample row and local Z is sample column. KangEngine
    // height grids are source row-major where col is X and row is horizontal Z
    // (or Y in Z-up mode), so we transpose into PhysX's row/column convention.
    const int hfRows = cols;
    const int hfCols = rows;
    std::vector<PxHeightFieldSample> samples(static_cast<size_t>(hfRows) *
                                             static_cast<size_t>(hfCols));

    for (int srcRow = 0; srcRow < rows; ++srcRow) {
        for (int srcCol = 0; srcCol < cols; ++srcCol) {
            const int hfRow = srcCol;
            const int hfCol =
                (upAxis == UpAxis::Z) ? (rows - 1 - srcRow) : srcRow;
            auto& sample = samples[static_cast<size_t>(hfRow * hfCols + hfCol)];
            const float h = heights[srcRow * cols + srcCol];
            const float scaled = h / outHeightScale;
            sample.height = static_cast<PxI16>(std::clamp(
                std::lround(scaled),
                static_cast<long>(std::numeric_limits<PxI16>::min()),
                static_cast<long>(std::numeric_limits<PxI16>::max())));
            sample.materialIndex0 = 0;
            sample.materialIndex1 = 0;
            sample.clearTessFlag();
        }
    }
    return samples;
}

#ifdef KANGENGINE_USE_CUDA
void checkCudaRuntime(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

CUcontext getCurrentCudaDriverContext() {
    using CuCtxGetCurrentFn = CUresult (*)(CUcontext*);
    void* libcuda = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!libcuda)
        throw std::runtime_error("failed to dlopen libcuda.so");
    auto* cuCtxGetCurrent =
        reinterpret_cast<CuCtxGetCurrentFn>(dlsym(libcuda, "cuCtxGetCurrent"));
    if (!cuCtxGetCurrent)
        throw std::runtime_error("failed to find cuCtxGetCurrent");

    CUcontext context = nullptr;
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context)
        throw std::runtime_error("failed to get current CUDA context");
    return context;
}
#endif

} // namespace

static PxFilterFlags
kangFilterShader(PxFilterObjectAttributes attributes0, PxFilterData filterData0,
                 PxFilterObjectAttributes attributes1, PxFilterData filterData1,
                 PxPairFlags& pairFlags, const void* constantBlock,
                 PxU32 constantBlockSize) {
    PxFilterFlags flags = PxDefaultSimulationFilterShader(
        attributes0, filterData0, attributes1, filterData1, pairFlags,
        constantBlock, constantBlockSize);

    if (filterData0.word3 != 0 && filterData1.word3 != 0 &&
        filterData0.word2 != filterData1.word2) {
        return PxFilterFlag::eSUPPRESS;
    }

    return flags;
}

static PxFilterFlags
contactReportFilterShader(PxFilterObjectAttributes attributes0,
                          PxFilterData filterData0,
                          PxFilterObjectAttributes attributes1,
                          PxFilterData filterData1, PxPairFlags& pairFlags,
                          const void* constantBlock, PxU32 constantBlockSize) {
    PxFilterFlags flags =
        kangFilterShader(attributes0, filterData0, attributes1, filterData1,
                         pairFlags, constantBlock, constantBlockSize);
    if (flags & PxFilterFlag::eSUPPRESS)
        return flags;

    pairFlags |= PxPairFlag::eNOTIFY_TOUCH_FOUND |
                 PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
                 PxPairFlag::eNOTIFY_CONTACT_POINTS;
    return flags;
}

class PhysicsWorld::ContactReportCallback : public PxSimulationEventCallback {
  public:
    explicit ContactReportCallback(std::vector<ContactPoint>& contacts)
        : _contacts(contacts) {}

    void onConstraintBreak(PxConstraintInfo* constraints,
                           PxU32 count) override {
        PX_UNUSED(constraints);
        PX_UNUSED(count);
    }
    void onWake(PxActor** actors, PxU32 count) override {
        PX_UNUSED(actors);
        PX_UNUSED(count);
    }
    void onSleep(PxActor** actors, PxU32 count) override {
        PX_UNUSED(actors);
        PX_UNUSED(count);
    }
    void onTrigger(PxTriggerPair* pairs, PxU32 count) override {
        PX_UNUSED(pairs);
        PX_UNUSED(count);
    }
    void onAdvance(const PxRigidBody* const* bodyBuffer,
                   const PxTransform* poseBuffer, const PxU32 count) override {
        PX_UNUSED(bodyBuffer);
        PX_UNUSED(poseBuffer);
        PX_UNUSED(count);
    }

    void onContact(const PxContactPairHeader& pairHeader,
                   const PxContactPair* pairs, PxU32 nbPairs) override {
        std::vector<PxContactPairPoint> contactPoints;

        for (PxU32 i = 0; i < nbPairs; ++i) {
            const PxU32 contactCount = pairs[i].contactCount;
            if (contactCount == 0)
                continue;

            contactPoints.resize(contactCount);
            pairs[i].extractContacts(contactPoints.data(), contactCount);

            for (PxU32 j = 0; j < contactCount; ++j) {
                const PxContactPairPoint& p = contactPoints[j];
                ContactPoint contact;
                contact.position = pxToGlm(p.position);
                contact.normal = pxToGlm(p.normal);
                contact.impulse = pxToGlm(p.impulse);
                contact.separation = p.separation;
                contact.actor0 = pairHeader.actors[0];
                contact.actor1 = pairHeader.actors[1];
                _contacts.push_back(contact);
            }
        }
    }

  private:
    std::vector<ContactPoint>& _contacts;
};

PhysicsWorld::PhysicsWorld(PhysicsConfig config) {
    _dt = config.dt;
    _upAxis = config.upAxis;
    _gravity = PxVec3(config.gravity[0], config.gravity[1], config.gravity[2]);
    _friction =
        PxVec3(config.friction[0], config.friction[1], config.friction[2]);

    _foundation =
        PxCreateFoundation(PX_PHYSICS_VERSION, _allocator, _errorCallback);
    _physics = PxCreatePhysics(PX_PHYSICS_VERSION, *_foundation,
                               PxTolerancesScale(), true);
    PxInitExtensions(*_physics, nullptr);

    PxSceneDesc sceneDesc(_physics->getTolerancesScale());
    sceneDesc.gravity = _gravity;
    _dispatcher = PxDefaultCpuDispatcherCreate(config.cpuDispatcherThreads);
    sceneDesc.cpuDispatcher = _dispatcher;
    sceneDesc.filterShader = config.filterShader;
    if (config.filterShader == PxDefaultSimulationFilterShader)
        sceneDesc.filterShader = kangFilterShader;
    if (config.enableContactReports) {
        _contactCallback = std::make_unique<ContactReportCallback>(_contacts);
        sceneDesc.simulationEventCallback = _contactCallback.get();
        if (config.filterShader == PxDefaultSimulationFilterShader)
            sceneDesc.filterShader = contactReportFilterShader;
    }
    sceneDesc.solverType = config.solverType;
    if (config.enableGPU) {
#ifdef __APPLE__
        fmt::print(
            "PhysX GPU is not available on Apple builds; using CPU PhysX.\n");
#else
        PxCudaContextManagerDesc cudaContextManagerDesc;
#ifdef KANGENGINE_USE_CUDA
        checkCudaRuntime(cudaSetDevice(0), "cudaSetDevice");
        // Force the CUDA runtime context to exist, then give
        // that exact driver context to PhysX instead of letting PhysX make one.
        checkCudaRuntime(cudaFree(nullptr), "cudaFree(0)");
        CUcontext cudaContext = getCurrentCudaDriverContext();
        cudaContextManagerDesc.ctx = &cudaContext;
#endif
        _cudaContextManager = PxCreateCudaContextManager(
            *_foundation, cudaContextManagerDesc, PxGetProfilerCallback());
        if (_cudaContextManager && !_cudaContextManager->contextIsValid()) {
            _cudaContextManager->release();
            _cudaContextManager = nullptr;
        }
        if (_cudaContextManager) {
            sceneDesc.cudaContextManager = _cudaContextManager;
            sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
            sceneDesc.flags |= PxSceneFlag::eENABLE_PCM;
            sceneDesc.gpuDynamicsConfig.tempBufferCapacity =
                config.gpuDynamics.tempBufferCapacity;
            sceneDesc.gpuDynamicsConfig.maxRigidContactCount =
                config.gpuDynamics.maxRigidContactCount;
            sceneDesc.gpuDynamicsConfig.maxRigidPatchCount =
                config.gpuDynamics.maxRigidPatchCount;
            sceneDesc.gpuDynamicsConfig.heapCapacity =
                config.gpuDynamics.heapCapacity;
            sceneDesc.gpuDynamicsConfig.foundLostPairsCapacity =
                config.gpuDynamics.foundLostPairsCapacity;
            sceneDesc.gpuDynamicsConfig.foundLostAggregatePairsCapacity =
                config.gpuDynamics.foundLostAggregatePairsCapacity;
            sceneDesc.gpuDynamicsConfig.totalAggregatePairsCapacity =
                config.gpuDynamics.totalAggregatePairsCapacity;
            sceneDesc.gpuDynamicsConfig.collisionStackSize =
                config.gpuDynamics.collisionStackSize;
            sceneDesc.gpuMaxNumPartitions = config.gpuDynamics.maxNumPartitions;
#ifdef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
            sceneDesc.flags |= PxSceneFlag::eENABLE_DIRECT_GPU_API;
#endif
            sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
            fmt::print("PhysX GPU is enabled (broadphase: gpu, compute: {}).\n",
                       sceneDesc.gpuComputeVersion);
        } else {
            fmt::print(
                "Failed to initialize PhysX CUDA context; using CPU PhysX.\n");
        }
#endif
    }

    _scene = _physics->createScene(sceneDesc);

    _material = _physics->createMaterial(
        _friction[0], _friction[1],
        _friction[2]); // staticFriction, dynamicFriction, restitution
    _material->setFrictionCombineMode(PxCombineMode::eMIN);
    fmt::print("PhysX is initialized.\n");
}

PhysicsWorld::~PhysicsWorld() {
    if (_scene)
        _scene->release();
    if (_dispatcher)
        _dispatcher->release();
#ifndef __APPLE__
    if (_cudaContextManager)
        _cudaContextManager->release();
#endif
    if (_material)
        _material->release();
    for (auto* heightField : _heightFields) {
        if (heightField)
            heightField->release();
    }
    _heightFields.clear();
    for (auto& [key, material] : _materialCache) {
        PX_UNUSED(key);
        if (material)
            material->release();
    }
    _materialCache.clear();
    _contactCallback.reset();
    PxCloseExtensions();
    if (_physics)
        _physics->release();
    if (_foundation)
        _foundation->release();
};

PxMaterial*
PhysicsWorld::materialForDesc(const Physics::PhysicsMaterialDesc& material) {
    if (!_physics)
        return _material;
    if (std::abs(material.staticFriction - _friction.x) <= 1e-6f &&
        std::abs(material.dynamicFriction - _friction.y) <= 1e-6f &&
        std::abs(material.restitution - _friction.z) <= 1e-6f)
        return _material;

    auto quantize = [](float value) {
        return static_cast<int>(std::round(value * 10000.0f));
    };
    const MaterialKey key{quantize(material.staticFriction),
                          quantize(material.dynamicFriction),
                          quantize(material.restitution)};
    auto it = _materialCache.find(key);
    if (it != _materialCache.end())
        return it->second;

    PxMaterial* created = _physics->createMaterial(material.staticFriction,
                                                   material.dynamicFriction,
                                                   material.restitution);
    created->setFrictionCombineMode(PxCombineMode::eMIN);
    _materialCache.emplace(key, created);
    return created;
}

PxShape* PhysicsWorld::createExclusiveShape(
    PxRigidActor& actor, const PxGeometry& geometry,
    const Physics::PhysicsMaterialDesc& material) {
    PxMaterial* shapeMat = materialForDesc(material);
    return PxRigidActorExt::createExclusiveShape(actor, geometry, *shapeMat);
}

void PhysicsWorld::step() {
    clearContacts();
    _scene->simulate(_dt); // _dt is already deltaTime (1/60)
    _scene->fetchResults(true);
}

void PhysicsWorld::fecthData() {
    // TODO: complete me
    PxU32 nbActors = _scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
}

void PhysicsWorld::addDefaultGround() {
    PxVec3 normal = (_upAxis == UpAxis::Z) ? PxVec3(0, 0, 1) : PxVec3(0, 1, 0);
    PxRigidStatic* groundPlane =
        PxCreatePlane(*_physics, PxPlane(normal, 0), *_material);
    _scene->addActor(*groundPlane);
    registerGroundActor(groundPlane);
}

void PhysicsWorld::registerGroundActor(const PxActor* actor) {
    if (actor)
        _groundActors.insert(actor);
}

void PhysicsWorld::unregisterGroundActor(const PxActor* actor) {
    if (actor)
        _groundActors.erase(actor);
}

bool PhysicsWorld::isGroundActor(const PxActor* actor) const {
    return actor && _groundActors.count(actor) != 0;
}

void PhysicsWorld::addBox(float x, float y, float z) {
    PxTransform boxPose(PxVec3(x, y, z));
    PxRigidDynamic* box = PxCreateDynamic(
        *_physics, boxPose, PxBoxGeometry(0.5f, 0.5f, 0.5f), *_material, 1.0f);
    _scene->addActor(*box);
}

PxRigidStatic* PhysicsWorld::createStaticBox(const glm::vec3& halfExtents,
                                             const glm::vec3& pos,
                                             const glm::quat& rot,
                                             bool registerAsGround) {
    PxTransform pose(PxVec3(pos.x, pos.y, pos.z),
                     PxQuat(rot.x, rot.y, rot.z, rot.w));
    PxRigidStatic* actor = _physics->createRigidStatic(pose);
    if (!actor)
        return nullptr;
    PxShape* shape = PxRigidActorExt::createExclusiveShape(
        *actor, PxBoxGeometry(halfExtents.x, halfExtents.y, halfExtents.z),
        *_material);
    applyRigidContactOffsets(shape, 0.02f, 0.0f);
    _scene->addActor(*actor);
    if (registerAsGround)
        registerGroundActor(actor);
    return actor;
}

PxRigidStatic* PhysicsWorld::createStaticHeightField(
    const float* heights, int rows, int cols, float horizontalScale,
    const Physics::PhysicsMaterialDesc& material, UpAxis upAxis, bool center,
    bool registerAsGround) {
    if (!heights || rows < 2 || cols < 2 || horizontalScale <= 0.f)
        return nullptr;

    float heightScale = 1.f;
    std::vector<PxHeightFieldSample> samples =
        makeHeightFieldSamples(heights, rows, cols, upAxis, heightScale);

    PxHeightFieldDesc desc;
    desc.nbRows = static_cast<PxU32>(cols);
    desc.nbColumns = static_cast<PxU32>(rows);
    desc.format = PxHeightFieldFormat::eS16_TM;
    desc.samples.data = samples.data();
    desc.samples.stride = sizeof(PxHeightFieldSample);
    if (!desc.isValid())
        return nullptr;

    PxHeightField* heightField =
        PxCreateHeightField(desc, _physics->getPhysicsInsertionCallback());
    if (!heightField)
        return nullptr;
    _heightFields.push_back(heightField);

    PxHeightFieldGeometry geometry(heightField, PxMeshGeometryFlags(),
                                   heightScale, horizontalScale,
                                   horizontalScale);
    if (!geometry.isValid())
        return nullptr;

    PxRigidStatic* actor = _physics->createRigidStatic(
        heightFieldPose(rows, cols, horizontalScale, upAxis, center));
    if (!actor)
        return nullptr;

    PxShape* shape = createExclusiveShape(*actor, geometry, material);
    applyRigidContactOffsets(shape, 0.02f, 0.0f);
    _scene->addActor(*actor);
    if (registerAsGround)
        registerGroundActor(actor);
    return actor;
}

PxRigidDynamic* PhysicsWorld::createDynamicBox(const glm::vec3& halfExtents,
                                               const glm::vec3& pos,
                                               const glm::quat& rot,
                                               float density) {
    PxTransform pose(PxVec3(pos.x, pos.y, pos.z),
                     PxQuat(rot.x, rot.y, rot.z, rot.w));
    PxRigidDynamic* actor = PxCreateDynamic(
        *_physics, pose,
        PxBoxGeometry(halfExtents.x, halfExtents.y, halfExtents.z), *_material,
        density);
    _scene->addActor(*actor);
    return actor;
}

PxRigidDynamic* PhysicsWorld::createDynamicSphere(float radius,
                                                  const glm::vec3& pos,
                                                  const glm::quat& rot,
                                                  float density) {
    PxTransform pose(PxVec3(pos.x, pos.y, pos.z),
                     PxQuat(rot.x, rot.y, rot.z, rot.w));
    PxRigidDynamic* actor = PxCreateDynamic(
        *_physics, pose, PxSphereGeometry(radius), *_material, density);
    _scene->addActor(*actor);
    return actor;
}

PxRigidDynamic* PhysicsWorld::createDynamicRigid(
    const Character::CharacterData& data, const glm::vec3& pos,
    const glm::quat& rot, float density, PxU32 collisionGroup,
    float contactOffset, float restOffset,
    const std::vector<Physics::CollisionMaterialOverride>& materialOverrides) {
    PxTransform pose(PxVec3(pos.x, pos.y, pos.z),
                     PxQuat(rot.x, rot.y, rot.z, rot.w));
    PxRigidDynamic* actor = _physics->createRigidDynamic(pose);
    if (!actor)
        return nullptr;

    const std::vector<Character::CollisionGeomDesc>* geoms = nullptr;
    int sourceBodyIndex = -1;
    auto rootIt = data.collisionGeoms.find(0);
    if (rootIt != data.collisionGeoms.end() && !rootIt->second.empty()) {
        geoms = &rootIt->second;
        sourceBodyIndex = 0;
    } else {
        for (const auto& [bodyIdx, bodyGeoms] : data.collisionGeoms) {
            if (!bodyGeoms.empty()) {
                geoms = &bodyGeoms;
                sourceBodyIndex = bodyIdx;
                break;
            }
        }
    }

    if (!geoms) {
        Character::CollisionGeomDesc fallbackGeom;
        fallbackGeom.name = "__fallback_sphere";
        const auto material = resolveCollisionMaterial(
            fallbackGeom, materialOverrides, data.skeletonTree, -1, 0);
        PxShape* shape =
            createExclusiveShape(*actor, PxSphereGeometry(0.1f), material);
        applyRigidContactOffsets(shape, contactOffset, restOffset);
    } else {
        using Type = Character::CollisionGeomDesc::Type;
        for (std::size_t i = 0; i < geoms->size(); ++i) {
            const auto& g = (*geoms)[i];
            const auto material = resolveCollisionMaterial(
                g, materialOverrides, data.skeletonTree, sourceBodyIndex,
                static_cast<int>(i));

            PxShape* shape = nullptr;
            PxTransform localPose(PxIdentity);
            switch (g.type) {
            case Type::Capsule:
            case Type::Cylinder: {
                float radius = g.size[0];
                if (g.hasFromTo) {
                    float halfH = (g.to - g.from).norm() * 0.5f;
                    shape = createExclusiveShape(
                        *actor, PxCapsuleGeometry(radius, halfH), material);
                    localPose = rigidFromToPose(g.from, g.to);
                } else {
                    shape = createExclusiveShape(
                        *actor, PxCapsuleGeometry(radius, g.size[1]), material);
                    localPose =
                        PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()),
                                    rigidMjcfShapeRot(g.quat));
                }
                break;
            }
            case Type::Sphere:
                shape = createExclusiveShape(
                    *actor, PxSphereGeometry(g.size[0]), material);
                localPose =
                    PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()));
                break;
            case Type::Box:
                shape = createExclusiveShape(
                    *actor, PxBoxGeometry(g.size[0], g.size[1], g.size[2]),
                    material);
                localPose = PxTransform(
                    PxVec3(g.pos.x(), g.pos.y(), g.pos.z()),
                    PxQuat(g.quat.x(), g.quat.y(), g.quat.z(), g.quat.w()));
                break;
            }

            if (shape) {
                shape->setLocalPose(localPose);
                applyRigidContactOffsets(
                    shape, g.margin >= 0.f ? g.margin : contactOffset,
                    restOffset);
            }
        }
    }

    PxRigidBodyExt::updateMassAndInertia(*actor, density);
    setRigidCollisionFilterData(actor, collisionGroup);
    _scene->addActor(*actor);
    return actor;
}

PxRigidStatic* PhysicsWorld::createStaticRigid(
    const Character::CharacterData& data, const glm::vec3& pos,
    const glm::quat& rot, PxU32 collisionGroup, float contactOffset,
    float restOffset,
    const std::vector<Physics::CollisionMaterialOverride>& materialOverrides) {
    PxTransform pose(PxVec3(pos.x, pos.y, pos.z),
                     PxQuat(rot.x, rot.y, rot.z, rot.w));
    PxRigidStatic* actor = _physics->createRigidStatic(pose);
    if (!actor)
        return nullptr;

    const std::vector<Character::CollisionGeomDesc>* geoms = nullptr;
    int sourceBodyIndex = -1;
    auto rootIt = data.collisionGeoms.find(0);
    if (rootIt != data.collisionGeoms.end() && !rootIt->second.empty()) {
        geoms = &rootIt->second;
        sourceBodyIndex = 0;
    } else {
        for (const auto& [bodyIdx, bodyGeoms] : data.collisionGeoms) {
            if (!bodyGeoms.empty()) {
                geoms = &bodyGeoms;
                sourceBodyIndex = bodyIdx;
                break;
            }
        }
    }

    if (!geoms) {
        Character::CollisionGeomDesc fallbackGeom;
        fallbackGeom.name = "__fallback_sphere";
        const auto material = resolveCollisionMaterial(
            fallbackGeom, materialOverrides, data.skeletonTree, -1, 0);
        PxShape* shape =
            createExclusiveShape(*actor, PxSphereGeometry(0.1f), material);
        applyRigidContactOffsets(shape, contactOffset, restOffset);
    } else {
        using Type = Character::CollisionGeomDesc::Type;
        for (std::size_t i = 0; i < geoms->size(); ++i) {
            const auto& g = (*geoms)[i];
            const auto material = resolveCollisionMaterial(
                g, materialOverrides, data.skeletonTree, sourceBodyIndex,
                static_cast<int>(i));

            PxShape* shape = nullptr;
            PxTransform localPose(PxIdentity);
            switch (g.type) {
            case Type::Capsule:
            case Type::Cylinder: {
                const float radius = g.size[0];
                if (g.hasFromTo) {
                    const float halfH = (g.to - g.from).norm() * 0.5f;
                    shape = createExclusiveShape(
                        *actor, PxCapsuleGeometry(radius, halfH), material);
                    localPose = rigidFromToPose(g.from, g.to);
                } else {
                    shape = createExclusiveShape(
                        *actor, PxCapsuleGeometry(radius, g.size[1]), material);
                    localPose =
                        PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()),
                                    rigidMjcfShapeRot(g.quat));
                }
                break;
            }
            case Type::Sphere:
                shape = createExclusiveShape(
                    *actor, PxSphereGeometry(g.size[0]), material);
                localPose =
                    PxTransform(PxVec3(g.pos.x(), g.pos.y(), g.pos.z()));
                break;
            case Type::Box:
                shape = createExclusiveShape(
                    *actor, PxBoxGeometry(g.size[0], g.size[1], g.size[2]),
                    material);
                localPose = PxTransform(
                    PxVec3(g.pos.x(), g.pos.y(), g.pos.z()),
                    PxQuat(g.quat.x(), g.quat.y(), g.quat.z(), g.quat.w()));
                break;
            }

            if (shape) {
                shape->setLocalPose(localPose);
                applyRigidContactOffsets(
                    shape, g.margin >= 0.f ? g.margin : contactOffset,
                    restOffset);
            }
        }
    }

    setRigidCollisionFilterData(actor, collisionGroup);
    _scene->addActor(*actor);
    return actor;
}

int PhysicsWorld::setRigidCollisionMaterial(
    PxRigidDynamic& rigid, const Physics::PhysicsMaterialDesc& material) {
    if (!_physics)
        return 0;

    const PxU32 shapeCount = rigid.getNbShapes();
    if (shapeCount == 0)
        return 0;

    int updated = 0;
    std::vector<PxShape*> shapes(shapeCount);
    rigid.getShapes(shapes.data(), shapeCount);
    for (auto* shape : shapes) {
        if (!shape)
            continue;
        PxMaterial* shapeMat = materialForDesc(material);
        shape->setMaterials(&shapeMat, 1);
        ++updated;
    }
    return updated;
}

int PhysicsWorld::setRigidCollisionMaterialOverrides(
    PxRigidDynamic& rigid, const Character::CharacterData& data,
    const std::vector<Physics::CollisionMaterialOverride>& overrides) {
    if (!_physics)
        return 0;

    const std::vector<Character::CollisionGeomDesc>* geoms = nullptr;
    int sourceBodyIndex = -1;
    auto rootIt = data.collisionGeoms.find(0);
    if (rootIt != data.collisionGeoms.end() && !rootIt->second.empty()) {
        geoms = &rootIt->second;
        sourceBodyIndex = 0;
    } else {
        for (const auto& [bodyIdx, bodyGeoms] : data.collisionGeoms) {
            if (!bodyGeoms.empty()) {
                geoms = &bodyGeoms;
                sourceBodyIndex = bodyIdx;
                break;
            }
        }
    }

    const PxU32 shapeCount = rigid.getNbShapes();
    if (shapeCount == 0)
        return 0;

    int updated = 0;
    std::vector<PxShape*> shapes(shapeCount);
    rigid.getShapes(shapes.data(), shapeCount);

    if (!geoms) {
        Character::CollisionGeomDesc fallbackGeom;
        fallbackGeom.name = "__fallback_sphere";
        const auto material = resolveCollisionMaterial(
            fallbackGeom, overrides, data.skeletonTree, -1, 0);
        for (auto* shape : shapes) {
            if (!shape)
                continue;
            PxMaterial* shapeMat = materialForDesc(material);
            shape->setMaterials(&shapeMat, 1);
            ++updated;
        }
        return updated;
    }

    const std::size_t count =
        std::min<std::size_t>(geoms->size(), shapes.size());
    for (std::size_t i = 0; i < count; ++i) {
        auto* shape = shapes[i];
        if (!shape)
            continue;
        const auto material =
            resolveCollisionMaterial((*geoms)[i], overrides, data.skeletonTree,
                                     sourceBodyIndex, static_cast<int>(i));
        PxMaterial* shapeMat = materialForDesc(material);
        shape->setMaterials(&shapeMat, 1);
        ++updated;
    }
    return updated;
}

std::vector<float>
PhysicsWorld::getContactForcesFlat(const Articulation& articulation,
                                   bool groundOnly) const {
    std::vector<float> out;
    const auto& links = articulation.links();
    out.assign(links.size() * 3, 0.0f);
    if (_contacts.empty() || _dt <= 0.0f)
        return out;

    std::unordered_map<const PxActor*, std::size_t> linkIndices;
    linkIndices.reserve(links.size());
    for (std::size_t i = 0; i < links.size(); ++i) {
        if (links[i])
            linkIndices.emplace(links[i], i);
    }

    auto addForce = [&out, &linkIndices](const PxActor* actor,
                                         const glm::vec3& force) {
        const auto it = linkIndices.find(actor);
        if (it == linkIndices.end())
            return;
        const std::size_t i = it->second;
        out[i * 3 + 0] += force.x;
        out[i * 3 + 1] += force.y;
        out[i * 3 + 2] += force.z;
    };

    for (const auto& contact : _contacts) {
        const bool actor0IsGround = isGroundActor(contact.actor0);
        const bool actor1IsGround = isGroundActor(contact.actor1);
        if (groundOnly && !actor0IsGround && !actor1IsGround)
            continue;

        const glm::vec3 force0 = contact.impulse / _dt;
        const glm::vec3 force1 = -force0;
        if (!groundOnly || actor1IsGround)
            addForce(contact.actor0, force0);
        if (!groundOnly || actor0IsGround)
            addForce(contact.actor1, force1);
    }
    return out;
}

std::vector<float>
PhysicsWorld::getRigidContactForceFlat(const PxRigidDynamic& rigid,
                                       bool groundOnly) const {
    std::vector<float> out(3, 0.0f);
    if (_contacts.empty() || _dt <= 0.0f)
        return out;

    const PxActor* rigidActor = &rigid;
    auto addForce = [&out](const glm::vec3& force) {
        out[0] += force.x;
        out[1] += force.y;
        out[2] += force.z;
    };

    for (const auto& contact : _contacts) {
        const bool actor0IsRigid = contact.actor0 == rigidActor;
        const bool actor1IsRigid = contact.actor1 == rigidActor;
        if (!actor0IsRigid && !actor1IsRigid)
            continue;

        const bool actor0IsGround = isGroundActor(contact.actor0);
        const bool actor1IsGround = isGroundActor(contact.actor1);
        if (groundOnly && !actor0IsGround && !actor1IsGround)
            continue;

        const glm::vec3 force0 = contact.impulse / _dt;
        const glm::vec3 force1 = -force0;
        if (actor0IsRigid && (!groundOnly || actor1IsGround))
            addForce(force0);
        if (actor1IsRigid && (!groundOnly || actor0IsGround))
            addForce(force1);
    }

    return out;
}

} // namespace KE
