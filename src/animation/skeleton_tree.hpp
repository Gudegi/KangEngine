#ifndef _SKELETON_TREE_HPP_
#define _SKELETON_TREE_HPP_

#include <Eigen/Core>
#include <string>
#include <unordered_map>
#include <vector>

// forward declaration
namespace tinyxml2 {
class XMLElement;
}

namespace KE {
namespace Animation {

class SkeletonTree {
  public:
    SkeletonTree() = default;
    SkeletonTree(std::vector<std::string> nodeNames,
                 std::vector<int> parentIndices,
                 std::vector<Eigen::Vector3f> localTranslations,
                 std::vector<int> numJointsInBody);

    // Parse MJCF file (opens and parses XML internally)
    static SkeletonTree skelFromMJCF(const std::string& path,
                                     const std::string& order = "DFS");

    // Parse from an already-open XML root element (avoids re-parsing the file)
    static SkeletonTree skelFromMJCFElement(tinyxml2::XMLElement* mjcfRoot,
                                            const std::string& order = "DFS");

    // Accessors
    int numJoints() const { return static_cast<int>(_nodeNames.size()); }
    const std::string& nodeName(int i) const { return _nodeNames[i]; }
    int parentIndex(int i) const { return _parentIndices[i]; }
    int index(const std::string& name) const;
    const Eigen::Vector3f& localTranslation(int i) const {
        return _localTranslations[i];
    }
    int numJointsInBody(int i) const { return _numJointsInBody[i]; }

    const std::vector<std::string>& nodeNames() const { return _nodeNames; }
    const std::vector<int>& parentIndices() const { return _parentIndices; }
    const std::vector<Eigen::Vector3f>& localTranslations() const {
        return _localTranslations;
    }

    // Debug print
    void print() const;

  private:
    std::vector<std::string> _nodeNames;
    std::vector<int> _parentIndices; // -1 for root
    std::vector<Eigen::Vector3f> _localTranslations;
    std::vector<int> _numJointsInBody;
    std::unordered_map<std::string, int> _nodeIndices;
};

} // namespace Animation
} // namespace KE

#endif
