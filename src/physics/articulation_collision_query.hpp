#pragma once

// Collects articulation and static environment shapes from a physics world,
// filters collision pairs, and checks batches of supplied link poses.
// Delegates each pair to Geometry::queryCollision and attaches frame/body/shape
// identifiers to results. Does not move actors or advance the simulation.
// Limits: discrete pose checks are not CCD and can miss collisions between poses.
// Inherits Geometry::queryCollision's approximation and containment limits.

#include "articulation.hpp"
#include "geometry/collision_query.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>

namespace KE::Physics {

// Query-only records; actor/shape pointers never escape a synchronous call.
struct QueryShape {
    PxGeometryHolder geometry;
    PxTransform localPose{PxIdentity};
    PxTransform actorPose{PxIdentity};
    int bodyIdx = -1;
    int shapeIdx = -1;
    int environmentIdx = -1;
    std::string name;
};

struct CollisionQueryResult {
    size_t frameIdx = 0;
    int bodyA = -1;
    int shapeA = -1;
    int bodyB = -1;
    int shapeB = -1;
    int environmentIdx = -1;
    std::string status;
    // NaN means unavailable, never a zero-depth assertion of safety.
    float depth = std::numeric_limits<float>::quiet_NaN();
    PxVec3 direction{std::numeric_limits<float>::quiet_NaN()};
};

struct CollisionQueryShapes {
    std::vector<QueryShape> robot, environment;
    std::set<std::array<int, 2>> adjacent;
    int numBodies = 0;
};

inline std::array<int, 2> orderedPair(int a, int b) {
    return {std::min(a, b), std::max(a, b)};
}

inline CollisionQueryShapes collectQueryShapes(
    PhysicsWorld& world, const Articulation& robot,
    const std::vector<PxRigidStatic*>& environment) {
    // Checks articulation/world lifetime before accessing links.
    const auto layout = robot.getCollisionShapes();
    const auto& links = robot.links();
    if (links.empty() || links[0]->getScene() != world.getScene())
        throw std::invalid_argument("Articulation must belong to this physics world");

    // Validate membership before dereferencing externally held actor handles.
    if (!environment.empty()) {
        const auto count = world.getScene()->getNbActors(PxActorTypeFlag::eRIGID_STATIC);
        std::vector<PxActor*> actors(count);
        world.getScene()->getActors(PxActorTypeFlag::eRIGID_STATIC, actors.data(), count);
        for (auto* actor : environment)
            if (!actor || std::find(actors.begin(), actors.end(), actor) == actors.end())
                throw std::invalid_argument("Environment actor must be a live static actor in this world");
    }

    CollisionQueryShapes result;
    result.numBodies = static_cast<int>(links.size());
    size_t layoutIdx = 0;
    auto append = [](const PxRigidActor& actor, int bodyIdx, int environmentIdx,
                     const std::string& name, std::vector<QueryShape>& destination) {
        std::vector<PxShape*> shapes(actor.getNbShapes());
        actor.getShapes(shapes.data(), static_cast<PxU32>(shapes.size()));
        for (size_t shapeIdx = 0; shapeIdx < shapes.size(); ++shapeIdx) {
            QueryShape record;
            record.geometry.storeAny(shapes[shapeIdx]->getGeometry());
            record.localPose = shapes[shapeIdx]->getLocalPose();
            record.actorPose = actor.getGlobalPose();
            record.bodyIdx = bodyIdx;
            record.shapeIdx = static_cast<int>(shapeIdx);
            record.environmentIdx = environmentIdx;
            record.name = name + ":" + std::to_string(shapeIdx);
            destination.push_back(record);
        }
    };
    for (int bodyIdx = 0; bodyIdx < result.numBodies; ++bodyIdx) {
        append(*links[bodyIdx], bodyIdx, -1, robot.bodyName(bodyIdx), result.robot);
        if (auto* joint = links[bodyIdx]->getInboundJoint()) {
            auto* parent = &joint->getParentArticulationLink();
            auto found = std::find(links.begin(), links.end(), parent);
            if (found != links.end())
                result.adjacent.insert(orderedPair(bodyIdx, static_cast<int>(found - links.begin())));
        }
    }
    for (auto& shape : result.robot)
        shape.name = layout.at(layoutIdx++).name;
    for (size_t idx = 0; idx < environment.size(); ++idx)
        append(*environment[idx], -1, static_cast<int>(idx), "environment_" + std::to_string(idx), result.environment);
    return result;
}

inline std::vector<CollisionQueryResult> queryCollisionPoses(
    const CollisionQueryShapes& shapes, const std::vector<PxTransform>& poses,
    size_t numFrames, bool selfCollision, bool excludeAdjacent,
    const std::vector<std::array<int, 2>>* includedBodyPairs,
    const std::vector<std::array<int, 2>>& excludedBodyPairs,
    const std::vector<std::array<int, 2>>& excludedEnvironmentPairs) {
    if (poses.size() != numFrames * static_cast<size_t>(shapes.numBodies))
        throw std::invalid_argument("Link pose count does not match frame/body dimensions");
    for (const auto& pose : poses)
        if (!pose.isValid())
            throw std::invalid_argument("Link poses require finite xyz and unit xyzw quaternions");
    auto bodyPairs = [&](const std::vector<std::array<int, 2>>& pairs) {
        std::set<std::array<int, 2>> result;
        for (auto pair : pairs) {
            if (pair[0] < 0 || pair[1] < 0 || pair[0] >= shapes.numBodies || pair[1] >= shapes.numBodies || pair[0] == pair[1])
                throw std::invalid_argument("Body pairs require two distinct valid body indices");
            result.insert(orderedPair(pair[0], pair[1]));
        }
        return result;
    };
    const auto excluded = bodyPairs(excludedBodyPairs);
    const auto included = includedBodyPairs ? bodyPairs(*includedBodyPairs)
                                          : std::set<std::array<int, 2>>{};
    std::set<std::array<int, 2>> excludedEnvironment;
    for (const auto& pair : excludedEnvironmentPairs) {
        bool exists = std::any_of(shapes.environment.begin(), shapes.environment.end(),
            [&](const auto& shape) { return shape.environmentIdx == pair[1]; });
        if (pair[0] < 0 || pair[0] >= shapes.numBodies || !exists)
            throw std::invalid_argument("Environment exclusions require valid body and environment indices");
        excludedEnvironment.insert(pair);
    }

    // Build filtered pairs once for the entire batch.
    // Pair indices address robot shapes first, followed by environment shapes.
    const size_t numRobotShapes = shapes.robot.size();
    std::vector<std::pair<size_t, size_t>> pairs;
    for (size_t a = 0; a < shapes.robot.size(); ++a) {
        const auto& first = shapes.robot[a];
        if (selfCollision) {
            for (size_t b = a + 1; b < shapes.robot.size(); ++b) {
                const auto& second = shapes.robot[b];
                const auto key = orderedPair(first.bodyIdx, second.bodyIdx);
                if (first.bodyIdx == second.bodyIdx || excluded.count(key))
                    continue;
                if (includedBodyPairs ? !included.count(key)
                                      : (excludeAdjacent && shapes.adjacent.count(key)))
                    continue;
                pairs.emplace_back(a, b);
            }
        }
        for (size_t b = 0; b < shapes.environment.size(); ++b)
            if (!excludedEnvironment.count({first.bodyIdx, shapes.environment[b].environmentIdx}))
                pairs.emplace_back(a, numRobotShapes + b);
    }

    std::vector<CollisionQueryResult> results;
    if (pairs.empty() || numFrames == 0)
        return results;

    // Static shape poses are shared by all frames in this call.
    std::vector<PxTransform> worldPoses(numRobotShapes + shapes.environment.size());
    for (size_t idx = 0; idx < shapes.environment.size(); ++idx) {
        const auto& shape = shapes.environment[idx];
        worldPoses[numRobotShapes + idx] = shape.actorPose * shape.localPose;
    }

    for (size_t frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
        // Each robot shape transform is composed once per frame.
        for (size_t idx = 0; idx < numRobotShapes; ++idx) {
            const auto& shape = shapes.robot[idx];
            worldPoses[idx] = poses[frameIdx * shapes.numBodies + shape.bodyIdx] * shape.localPose;
        }

        for (const auto& pair : pairs) {
            const auto& a = shapes.robot[pair.first];
            const auto& b = pair.second < numRobotShapes
                ? shapes.robot[pair.second] : shapes.environment[pair.second - numRobotShapes];
            const auto collision = Geometry::queryCollision(
                a.geometry.any(), worldPoses[pair.first], b.geometry.any(), worldPoses[pair.second]);
            if (!collision)
                continue;

            CollisionQueryResult row;
            row.frameIdx = frameIdx;
            row.bodyA = a.bodyIdx;
            row.shapeA = a.shapeIdx;
            row.bodyB = b.bodyIdx;
            row.shapeB = b.shapeIdx;
            row.environmentIdx = b.environmentIdx;
            row.status = collision->status;
            row.depth = collision->depth;
            row.direction = collision->direction;
            results.push_back(row);
        }
    }
    return results;
}

} // namespace KE::Physics
