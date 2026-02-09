#include "skeleton_tree.hpp"

#include <fmt/core.h>
#include <tinyxml2.h>

#include <queue>
#include <sstream>
#include <stdexcept>

namespace KE {
namespace Animation {

SkeletonTree::SkeletonTree(std::vector<std::string> nodeNames,
                           std::vector<int> parentIndices,
                           std::vector<Eigen::Vector3f> localTranslations,
                           std::vector<int> numJointsInBody)
    : _nodeNames(std::move(nodeNames)),
      _parentIndices(std::move(parentIndices)),
      _localTranslations(std::move(localTranslations)),
      _numJointsInBody(std::move(numJointsInBody)) {
    for (int i = 0; i < static_cast<int>(_nodeNames.size()); ++i) {
        _nodeIndices[_nodeNames[i]] = i;
    }
}

int SkeletonTree::index(const std::string& name) const {
    auto it = _nodeIndices.find(name);
    if (it == _nodeIndices.end()) {
        throw std::runtime_error(
            fmt::format("Node '{}' not found in skeleton tree", name));
    }
    return it->second;
}

// Parse "x y z" string into Vector3f
static Eigen::Vector3f parsePos(const char* posStr) {
    if (!posStr)
        return Eigen::Vector3f::Zero();
    float x = 0, y = 0, z = 0;
    std::istringstream ss(posStr);
    ss >> x >> y >> z;
    return Eigen::Vector3f(x, y, z);
}

SkeletonTree SkeletonTree::fromMJCF(const std::string& path) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error(
            fmt::format("Failed to load MJCF file: {}", path));
    }

    auto* root = doc.RootElement();
    auto* worldbody = root->FirstChildElement("worldbody");
    if (!worldbody) {
        throw std::runtime_error("MJCF: missing <worldbody> element");
    }

    auto* bodyRoot = worldbody->FirstChildElement("body");
    if (!bodyRoot) {
        throw std::runtime_error("MJCF: missing root <body> element");
    }

    // BFS traversal (matching my_poselib's from_mjcf)
    std::vector<std::string> nodeNames;
    std::vector<int> parentIndices;
    std::vector<Eigen::Vector3f> localTranslations;
    std::vector<int> numJointsInBody;

    struct BFSEntry {
        tinyxml2::XMLElement* element;
        int parentIndex;
        int currentIndex;
    };

    std::queue<BFSEntry> queue;
    int nodeIndex = 0;
    queue.push({bodyRoot, -1, nodeIndex});

    while (!queue.empty()) {
        auto [element, parentIdx, currentIdx] = queue.front();
        queue.pop();

        const char* name = element->Attribute("name");
        nodeNames.push_back(name ? name : fmt::format("body_{}", currentIdx));
        parentIndices.push_back(parentIdx);
        localTranslations.push_back(parsePos(element->Attribute("pos")));

        // Count joint children
        int jointCount = 0;
        for (auto* joint = element->FirstChildElement("joint"); joint;
             joint = joint->NextSiblingElement("joint")) {
            ++jointCount;
        }
        // Also count freejoint
        for (auto* fj = element->FirstChildElement("freejoint"); fj;
             fj = fj->NextSiblingElement("freejoint")) {
            ++jointCount;
        }
        numJointsInBody.push_back(jointCount);

        // Enqueue child bodies
        for (auto* child = element->FirstChildElement("body"); child;
             child = child->NextSiblingElement("body")) {
            ++nodeIndex;
            queue.push({child, currentIdx, nodeIndex});
        }
    }

    return SkeletonTree(std::move(nodeNames), std::move(parentIndices),
                        std::move(localTranslations),
                        std::move(numJointsInBody));
}

void SkeletonTree::print() const {
    fmt::print("SkeletonTree ({} joints):\n", numJoints());
    for (int i = 0; i < numJoints(); ++i) {
        const auto& t = _localTranslations[i];
        fmt::print("  [{}] {} (parent={}, pos=({:.3f}, {:.3f}, {:.3f}), "
                   "joints={})\n",
                   i, _nodeNames[i], _parentIndices[i], t.x(), t.y(), t.z(),
                   _numJointsInBody[i]);
    }
}

} // namespace Animation
} // namespace KE
