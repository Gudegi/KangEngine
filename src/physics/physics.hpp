///
/// Author Kyungwon Kang, 2025/04
///

#ifndef _PHYSICS_HPP_
#define _PHYSICS_HPP_

#include "PxPhysicsAPI.h"
#include <fmt/base.h>
#include "physics/collision/convex_collision.hpp"
#include "utils/types.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace physx;

namespace KE {

class Articulation;
namespace Asset {
struct ArticulationDesc;
} // namespace Asset
namespace Scene {
struct MeshData;
} // namespace Scene
namespace Physics {
struct CollisionMaterialOverride;
struct PhysicsMaterialDesc;

// Reconstruct a render mesh from the exact cooked PhysX convex geometry. The
// shape-local pose is baked into the returned actor/link-local vertices.
std::shared_ptr<Scene::MeshData>
buildConvexCollisionMesh(const physx::PxShape& shape);
} // namespace Physics

struct PhysicsGpuDynamicsConfig {
    uint64_t tempBufferCapacity = 64ull * 1024 * 1024;
    uint32_t maxRigidContactCount = 4u * 1024 * 1024;
    uint32_t maxRigidPatchCount = 512u * 1024;
    uint32_t heapCapacity = 256u * 1024 * 1024;
    uint32_t foundLostPairsCapacity = 4u * 1024 * 1024;
    uint32_t foundLostAggregatePairsCapacity = 32u * 1024 * 1024;
    uint32_t totalAggregatePairsCapacity = 2u * 1024 * 1024;
    uint32_t collisionStackSize = 256u * 1024 * 1024;
    uint32_t maxNumPartitions = 8;
};

struct PhysicsConfig {
    UpAxis upAxis = UpAxis::Y;
    float dt = 1.0f / 60.0f;
    float gravity[3] = {0.0f, -9.81f, 0.0f};
    float friction[3] = {1.0f, 1.0f, 0.0f};
    PxSimulationFilterShader filterShader = PxDefaultSimulationFilterShader;
    bool enableContactReports = true;
    bool enableBodyAccelerations = false;
    uint32_t cpuDispatcherThreads = 4;
    // PxSolverType::Enum solverType = PxSolverType::ePGS;
    PxSolverType::Enum solverType = PxSolverType::eTGS;
    float bounceThresholdVelocity = 2.0f;
    float frictionOffsetThreshold = 0.04f;
    float frictionCorrelationDistance = 0.025f;
    bool enableStabilization = false;
    PhysicsGpuDynamicsConfig gpuDynamics;
    bool enableGPU = false;

    static PhysicsConfig yUp() { return {}; }

    static PhysicsConfig zUp() {
        PhysicsConfig c;
        c.upAxis = UpAxis::Z;
        c.gravity[1] = 0.f;
        c.gravity[2] = -9.81f;
        return c;
    }
};

struct ContactPoint {
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f);
    glm::vec3 impulse = glm::vec3(0.0f);
    float separation = 0.0f;
    PxActor* actor0 = nullptr;
    PxActor* actor1 = nullptr;
};

class PhysicsWorld {

  private:
    class ContactReportCallback;

    struct MaterialKey {
        int staticFriction = 0;
        int dynamicFriction = 0;
        int restitution = 0;

        bool operator==(const MaterialKey& other) const {
            return staticFriction == other.staticFriction &&
                   dynamicFriction == other.dynamicFriction &&
                   restitution == other.restitution;
        }
    };

    struct MaterialKeyHash {
        std::size_t operator()(const MaterialKey& key) const {
            const std::size_t a = static_cast<std::size_t>(key.staticFriction);
            const std::size_t b = static_cast<std::size_t>(key.dynamicFriction);
            const std::size_t c = static_cast<std::size_t>(key.restitution);
            return a ^ (b + 0x9e3779b9u + (a << 6) + (a >> 2)) ^
                   (c + 0x9e3779b9u + (b << 6) + (b >> 2));
        }
    };

    PxDefaultAllocator _allocator;
    PxDefaultErrorCallback _errorCallback;
    PxFoundation* _foundation = nullptr;
    PxPhysics* _physics = nullptr;
    PxScene* _scene = nullptr;
    PxMaterial* _material = nullptr;
    PxDefaultCpuDispatcher* _dispatcher = nullptr;
    PxCudaContextManager* _cudaContextManager = nullptr;
    std::vector<physx::PxHeightField*> _heightFields;
    std::vector<physx::PxConvexMesh*> _convexMeshes;
    struct CachedConvexMesh {
        std::shared_ptr<const Scene::MeshData> source;
        Physics::ConvexCookingOptions options;
        physx::PxConvexMesh* mesh = nullptr;
    };
    std::vector<CachedConvexMesh> _convexMeshCache;
    std::shared_ptr<Physics::PhysicsResourceLifetimeToken>
        _resourceLifetimeToken =
            std::make_shared<Physics::PhysicsResourceLifetimeToken>();

    float _dt;
    UpAxis _upAxis;
    PxVec3 _gravity;
    PxVec3 _friction;
    std::vector<ContactPoint> _contacts;
    std::unordered_set<const PxActor*> _groundActors;
    std::unordered_map<MaterialKey, PxMaterial*, MaterialKeyHash>
        _materialCache;
    std::unique_ptr<ContactReportCallback> _contactCallback;

    physx::PxMaterial*
    materialForDesc(const Physics::PhysicsMaterialDesc& material);
  public:
    PhysicsWorld(PhysicsConfig config);
    ~PhysicsWorld();

    void setDt(float dt) { _dt = dt; }

    void addDefaultGround();
    void registerGroundActor(const PxActor* actor);
    void unregisterGroundActor(const PxActor* actor);
    void clearGroundActors() { _groundActors.clear(); }
    bool isGroundActor(const PxActor* actor) const;
    PxU32 numGroundActors() const {
        return static_cast<PxU32>(_groundActors.size());
    }
    PxU32 numCachedMaterials() const {
        return static_cast<PxU32>(_materialCache.size());
    }

    // PhysX shapes are intentionally created as exclusive per actor/link
    // objects. Per-instance PxShape state (local pose, filter data, contact
    // offsets, material slot) stays local, while heavier resources such as
    // PxMaterial and cooked collision meshes are shared/cached by
    // PhysicsWorld.
    physx::PxShape*
    createExclusiveShape(physx::PxRigidActor& actor,
                         const physx::PxGeometry& geometry,
                         const Physics::PhysicsMaterialDesc& material);

    bool attachConvexCollision(
        physx::PxRigidActor& actor,
        const Physics::ConvexCollisionResource& collision,
        const Physics::PhysicsMaterialDesc& material,
        const glm::vec3& localPosition = glm::vec3(0.0f),
        const glm::quat& localRotation =
            glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        float contactOffset = 0.02f, float restOffset = 0.0f);

    void setRigidCollisionGroup(physx::PxRigidActor& actor,
                                PxU32 collisionGroup);
    void addRigidActor(physx::PxRigidActor& actor);
    void destroyRigidActor(physx::PxRigidActor* actor);

    // Cook once per shared source mesh/options pair and reuse the heavy PhysX
    // convex resource across exclusive shape instances.
    physx::PxConvexMesh*
    getOrCreateConvexMesh(std::shared_ptr<const Scene::MeshData> mesh,
                          const Physics::ConvexCookingOptions& cooking = {});

    std::shared_ptr<Physics::ConvexCollisionResource>
    createConvexCollision(const std::vector<Physics::ConvexMeshPart>& parts,
                          const Physics::ConvexCookingOptions& cooking = {});

    physx::PxRigidDynamic* createDynamicFromCollision(
        const std::shared_ptr<Physics::ConvexCollisionResource>& collision,
        const glm::vec3& pos, const glm::quat& rot, float density,
        const Physics::PhysicsMaterialDesc& material, PxU32 collisionGroup = 0,
        float contactOffset = 0.02f, float restOffset = 0.0f);

    physx::PxRigidStatic* createStaticFromCollision(
        const std::shared_ptr<Physics::ConvexCollisionResource>& collision,
        const glm::vec3& pos, const glm::quat& rot,
        const Physics::PhysicsMaterialDesc& material, PxU32 collisionGroup = 0,
        float contactOffset = 0.02f, float restOffset = 0.0f,
        bool registerAsGround = false);

    physx::PxRigidDynamic* createDynamicConvexCompound(
        const std::vector<Physics::ConvexMeshPart>& parts, const glm::vec3& pos,
        const glm::quat& rot, float density,
        const Physics::ConvexCookingOptions& cooking,
        const Physics::PhysicsMaterialDesc& material, PxU32 collisionGroup = 0,
        float contactOffset = 0.02f, float restOffset = 0.0f);

    physx::PxRigidStatic* createStaticConvexCompound(
        const std::vector<Physics::ConvexMeshPart>& parts, const glm::vec3& pos,
        const glm::quat& rot, const Physics::ConvexCookingOptions& cooking,
        const Physics::PhysicsMaterialDesc& material, PxU32 collisionGroup = 0,
        float contactOffset = 0.02f, float restOffset = 0.0f,
        bool registerAsGround = false);

    void addBox(float x, float y, float z);
    physx::PxRigidStatic*
    createStaticBox(const glm::vec3& halfExtents, const glm::vec3& pos,
                    const glm::quat& rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                    bool registerAsGround = true);
    physx::PxRigidStatic* createStaticHeightField(
        const float* heights, int rows, int cols, float horizontalScale,
        const Physics::PhysicsMaterialDesc& material, UpAxis upAxis = UpAxis::Y,
        bool center = true, bool registerAsGround = true);

    physx::PxRigidDynamic*
    createDynamicBox(const glm::vec3& halfExtents, const glm::vec3& pos,
                     const glm::quat& rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                     float density = 1.0f);

    physx::PxRigidDynamic* createDynamicSphere(
        float radius, const glm::vec3& pos,
        const glm::quat& rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        float density = 1.0f);

    physx::PxRigidDynamic*
    createDynamicRigid(const Asset::ArticulationDesc& data,
                       const glm::vec3& pos,
                       const glm::quat& rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                       float density = 1.0f, PxU32 collisionGroup = 0,
                       float contactOffset = 0.02f, float restOffset = 0.0f,
                       const std::vector<Physics::CollisionMaterialOverride>&
                           materialOverrides = {});

    physx::PxRigidStatic*
    createStaticRigid(const Asset::ArticulationDesc& data, const glm::vec3& pos,
                      const glm::quat& rot = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                      PxU32 collisionGroup = 0, float contactOffset = 0.02f,
                      float restOffset = 0.0f,
                      const std::vector<Physics::CollisionMaterialOverride>&
                          materialOverrides = {});

    int setRigidCollisionMaterial(physx::PxRigidDynamic& rigid,
                                  const Physics::PhysicsMaterialDesc& material);

    int setRigidCollisionMaterialOverrides(
        physx::PxRigidDynamic& rigid, const Asset::ArticulationDesc& data,
        const std::vector<Physics::CollisionMaterialOverride>& overrides);

    void fecthData();

    void step();

    const std::vector<ContactPoint>& getContacts() const { return _contacts; }
    PxU32 numContacts() const { return static_cast<PxU32>(_contacts.size()); }
    void clearContacts() { _contacts.clear(); }
    std::vector<float> getContactForcesFlat(const Articulation& articulation,
                                            bool groundOnly = false) const;
    std::vector<float>
    getGroundContactForcesFlat(const Articulation& articulation) const {
        return getContactForcesFlat(articulation, true);
    }
    std::vector<float>
    getRigidContactForceFlat(const physx::PxRigidDynamic& rigid,
                             bool groundOnly = false) const;
    std::vector<float>
    getRigidGroundContactForceFlat(const physx::PxRigidDynamic& rigid) const {
        return getRigidContactForceFlat(rigid, true);
    }

    PxU32 numBodyActors() {
        return _scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    }

    UpAxis getUpAxis() const { return _upAxis; }
    PxPhysics* getPhysics() { return _physics; }
    PxMaterial* getMaterial() { return _material; }
    PxMaterial*
    getMaterialForDesc(const Physics::PhysicsMaterialDesc& material) {
        return materialForDesc(material);
    }
    PxScene* getScene() { return _scene; }
    const PxScene* getScene() const { return _scene; }
    PxCudaContextManager* getCudaContextManager() {
        return _cudaContextManager;
    }
    const PxCudaContextManager* getCudaContextManager() const {
        return _cudaContextManager;
    }
    bool isGpuEnabled() const { return _cudaContextManager != nullptr; }
    std::weak_ptr<Physics::PhysicsResourceLifetimeToken> lifetimeToken() const {
        return _resourceLifetimeToken;
    }
};

// PhysX > GLM conversion
inline glm::vec3 pxToGlm(const PxVec3& v) { return glm::vec3(v.x, v.y, v.z); }
inline glm::quat pxToGlm(const PxQuat& q) {
    return glm::quat(q.w, q.x, q.y, q.z);
}
inline glm::mat4 pxToMat4(const PxTransform& t) {
    glm::mat4 m = glm::mat4_cast(pxToGlm(t.q));
    m[3] = glm::vec4(pxToGlm(t.p), 1.f);
    return m;
}

inline glm::vec3 pxToVec3(const physx::PxVec3& v) {
    return glm::vec3(v.x, v.y, v.z);
}

inline physx::PxVec3 toPxVec3(const glm::vec3& v) {
    return physx::PxVec3(v.x, v.y, v.z);
}

} // namespace KE

#endif
