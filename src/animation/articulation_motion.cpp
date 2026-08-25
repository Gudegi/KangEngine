#include "articulation_motion.hpp"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace KE {
namespace Animation {

namespace {

// Return q and qd widths for one canonical coordinate block.
std::pair<int, int> coordinateSizes(ArticulationCoordinateType type) {
    switch (type) {
    case ArticulationCoordinateType::Fixed:
        return {0, 0};
    case ArticulationCoordinateType::Revolute:
    case ArticulationCoordinateType::Prismatic:
        return {1, 1};
    case ArticulationCoordinateType::Spherical:
        return {4, 3};
    case ArticulationCoordinateType::Free:
        return {7, 6};
    }
    throw std::runtime_error("Unsupported articulation coordinate type");
}

// Map an authored asset joint type to its canonical coordinate type.
ArticulationCoordinateType coordinateType(Asset::JointDesc::Type type) {
    switch (type) {
    case Asset::JointDesc::Type::Fixed:
        return ArticulationCoordinateType::Fixed;
    case Asset::JointDesc::Type::Revolute:
        return ArticulationCoordinateType::Revolute;
    case Asset::JointDesc::Type::Spherical:
        return ArticulationCoordinateType::Spherical;
    }
    throw std::runtime_error("Unsupported articulation joint type");
}

// Hash layout-defining metadata for compatibility checks.
std::string signatureFor(const ArticulationCoordinateLayout& layout) {
    std::ostringstream source;
    source << std::setprecision(9) << layout.traversalOrder() << '|'
           << layout.hasFreeRoot();
    for (const auto& block : layout.blocks()) {
        source << '|' << block.bodyIndex << ':' << block.parentBodyIndex << ':'
               << block.jointIndexInBody << ':' << block.bodyName << ':'
               << block.jointName << ':' << static_cast<int>(block.type) << ':'
               << block.axis.x() << ',' << block.axis.y() << ','
               << block.axis.z() << ':' << block.referenceTranslation.x() << ','
               << block.referenceTranslation.y() << ','
               << block.referenceTranslation.z() << ':'
               << block.referenceRotation.w() << ','
               << block.referenceRotation.x() << ','
               << block.referenceRotation.y() << ','
               << block.referenceRotation.z() << ':' << block.lowerLimit << ':'
               << block.upperLimit << ':' << block.qSize << ':' << block.qdSize;
    }

    uint64_t hash = 14695981039346656037ull;
    for (const unsigned char byte : source.str()) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    std::ostringstream result;
    result << std::hex << std::setfill('0') << std::setw(16) << hash;
    return result.str();
}

} // namespace

// Build a canonical layout without reordering source articulation bodies.
ArticulationCoordinateLayout
ArticulationCoordinateLayout::fromData(const Asset::ArticulationDesc& data,
                                       bool freeRoot) {
    if (!data.skeletonTree)
        throw std::runtime_error(
            "ArticulationCoordinateLayout requires a SkeletonTree");

    ArticulationCoordinateLayout layout;
    layout._skeletonTree = data.skeletonTree;
    layout._hasFreeRoot = freeRoot;
    layout._traversalOrder = data.traversalOrder;

    auto append = [&](int bodyIndex, int parentBodyIndex, int jointIndexInBody,
                      const std::string& bodyName, const std::string& jointName,
                      ArticulationCoordinateType type,
                      const Eigen::Vector3f& axis, float lowerLimit,
                      float upperLimit) {
        const auto [qSize, qdSize] = coordinateSizes(type);
        ArticulationCoordinateBlock block;
        block.bodyIndex = bodyIndex;
        block.parentBodyIndex = parentBodyIndex;
        block.jointIndexInBody = jointIndexInBody;
        block.bodyName = bodyName;
        block.jointName = jointName;
        block.type = type;
        block.axis = axis;
        block.referenceTranslation =
            data.skeletonTree->localTranslation(bodyIndex);
        block.referenceRotation = data.skeletonTree->localRotation(bodyIndex);
        block.lowerLimit = lowerLimit;
        block.upperLimit = upperLimit;
        block.qOffset = layout._nq;
        block.qSize = qSize;
        block.qdOffset = layout._nv;
        block.qdSize = qdSize;
        layout._blocks.push_back(std::move(block));
        layout._nq += qSize;
        layout._nv += qdSize;
    };

    if (freeRoot) {
        append(0, -1, -1, data.skeletonTree->nodeName(0), "$free_root",
               ArticulationCoordinateType::Free, Eigen::Vector3f::Zero(),
               -std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity());
    }

    // SkeletonTree node order is already the BFS/DFS order selected by the
    // loader. Iterate indices directly; rebuilding a traversal here would
    // silently change the articulation coordinate layout.
    for (int bodyIndex = 0; bodyIndex < data.skeletonTree->numJoints();
         ++bodyIndex) {
        const auto found = data.joints.find(bodyIndex);
        if (found == data.joints.end())
            continue;
        const auto& joints = found->second;
        for (int jointIndex = 0; jointIndex < static_cast<int>(joints.size());
             ++jointIndex) {
            const auto& joint = joints[static_cast<size_t>(jointIndex)];
            append(bodyIndex, data.skeletonTree->parentIndex(bodyIndex),
                   jointIndex, data.skeletonTree->nodeName(bodyIndex),
                   joint.name, coordinateType(joint.type), joint.axis,
                   joint.loLimit, joint.hiLimit);
        }
    }

    layout._modelSignature = signatureFor(layout);
    return layout;
}

ArticulationMotion::ArticulationMotion(ArticulationCoordinateLayout layout,
                                       int numFrames, float fps,
                                       std::string motionName,
                                       std::vector<float> q,
                                       std::vector<float> qd)
    : _layout(std::move(layout)), _numFrames(numFrames), _fps(fps),
      _motionName(std::move(motionName)), _q(std::move(q)), _qd(std::move(qd)) {
    if (_numFrames < 0)
        throw std::runtime_error(
            "ArticulationMotion frame count must be non-negative");
    if (_fps <= 0.0f)
        throw std::runtime_error("ArticulationMotion fps must be positive");
    if (_q.size() !=
        static_cast<size_t>(_numFrames) * static_cast<size_t>(_layout.nq()))
        throw std::runtime_error(
            "ArticulationMotion q shape does not match layout");
    if (_qd.size() !=
        static_cast<size_t>(_numFrames) * static_cast<size_t>(_layout.nv()))
        throw std::runtime_error(
            "ArticulationMotion qd shape does not match layout");
}

// Return the time from the first through the last sample.
float ArticulationMotion::duration() const {
    return _numFrames > 0 ? static_cast<float>(_numFrames - 1) / _fps : 0.0f;
}

} // namespace Animation
} // namespace KE
