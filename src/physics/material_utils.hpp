#pragma once

#include "PxPhysicsAPI.h"
#include "physics_material.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace KE::Physics {

inline physx::PxCombineMode::Enum toPhysX(MaterialCombineMode mode) {
    switch (mode) {
    case MaterialCombineMode::Average: return physx::PxCombineMode::eAVERAGE;
    case MaterialCombineMode::Min: return physx::PxCombineMode::eMIN;
    case MaterialCombineMode::Multiply: return physx::PxCombineMode::eMULTIPLY;
    case MaterialCombineMode::Max: return physx::PxCombineMode::eMAX;
    }
    throw std::invalid_argument("unknown material combine mode");
}

inline MaterialCombineMode fromPhysX(physx::PxCombineMode::Enum mode) {
    switch (mode) {
    case physx::PxCombineMode::eAVERAGE: return MaterialCombineMode::Average;
    case physx::PxCombineMode::eMIN: return MaterialCombineMode::Min;
    case physx::PxCombineMode::eMULTIPLY: return MaterialCombineMode::Multiply;
    case physx::PxCombineMode::eMAX: return MaterialCombineMode::Max;
    default: throw std::runtime_error("unknown PhysX material combine mode");
    }
}

inline void validateMaterial(const PhysicsMaterialDesc& material) {
    // Keep cache quantization representable and reject values PhysX would ignore.
    constexpr double maxFriction = (std::numeric_limits<int>::max() - 1.) / 10000.;
    for (float friction : {material.staticFriction, material.dynamicFriction}) {
        if (!std::isfinite(friction) || friction < 0.f || friction > maxFriction)
            throw std::invalid_argument("friction must be finite, non-negative, and below 214748");
    }
    if (!std::isfinite(material.restitution) || material.restitution < 0.f ||
        material.restitution > 1.f)
        throw std::invalid_argument("restitution must be finite and in [0, 1]");
    toPhysX(material.frictionCombineMode);
    toPhysX(material.restitutionCombineMode);
}

inline PhysicsMaterialDesc materialDesc(const physx::PxMaterial& material) {
    return {material.getStaticFriction(), material.getDynamicFriction(),
            material.getRestitution(), fromPhysX(material.getFrictionCombineMode()),
            fromPhysX(material.getRestitutionCombineMode())};
}

inline std::vector<PhysicsMaterialDesc> collisionMaterials(const physx::PxRigidActor& actor) {
    std::vector<physx::PxShape*> shapes(actor.getNbShapes());
    actor.getShapes(shapes.data(), static_cast<physx::PxU32>(shapes.size()));
    std::vector<PhysicsMaterialDesc> result;
    result.reserve(shapes.size());
    for (const auto* shape : shapes) {
        physx::PxMaterial* material = nullptr;
        if (shape->getNbMaterials() != 1 || shape->getMaterials(&material, 1) != 1 || !material)
            throw std::runtime_error("expected one collision material per shape");
        result.push_back(materialDesc(*material));
    }
    return result;
}

} // namespace KE::Physics
