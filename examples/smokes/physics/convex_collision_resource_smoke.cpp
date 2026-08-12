#include "physics/collision/convex_collision.hpp"
#include "physics/physics.hpp"
#include "physics/physics_material.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace KE;
using namespace KE::Physics;
using namespace physx;

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

PxConvexMesh* convexMeshOf(PxRigidActor& actor) {
    require(actor.getNbShapes() == 1, "expected one convex shape");
    PxShape* shape = nullptr;
    actor.getShapes(&shape, 1);
    require(shape != nullptr, "actor returned a null shape");
    const PxGeometryHolder geometry = shape->getGeometry();
    require(geometry.getType() == PxGeometryType::eCONVEXMESH,
            "shape is not a convex mesh");
    return geometry.convexMesh().convexMesh;
}

} // namespace

int main() {
    std::shared_ptr<ConvexCollisionResource> collision;
    {
        PhysicsWorld world(PhysicsConfig::zUp());

        ConvexMeshPart tetrahedron;
        tetrahedron.vertices = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
        };
        tetrahedron.indices = {
            0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3,
        };

        collision = world.createConvexCollision({tetrahedron});
        require(collision->isValid(), "new collision resource is invalid");
        require(collision->partCount() == 1,
                "collision resource has the wrong part count");

        PhysicsMaterialDesc material;
        PxRigidDynamic* first = world.createDynamicFromCollision(
            collision, {-1.0f, 0.0f, 2.0f}, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
            1.0f, material);
        PxRigidDynamic* second = world.createDynamicFromCollision(
            collision, {1.0f, 0.0f, 2.0f}, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
            1.0f, material);
        require(first && second,
                "failed to create actors from collision resource");
        require(convexMeshOf(*first) == convexMeshOf(*second),
                "actors did not reuse the cooked PxConvexMesh");
    }

    require(!collision->isValid(),
            "collision resource remained valid after PhysicsWorld destruction");
    std::cout << "convex_collision_resource_smoke: PASS\n";
    return 0;
}
