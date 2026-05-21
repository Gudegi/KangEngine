#ifndef _SKINNING_HPP_
#define _SKINNING_HPP_

#include <Eigen/Core>
#include <glm/mat4x4.hpp>

#include <array>
#include <vector>

namespace KE {
namespace Animation {

struct CPUSkinningInput {
    const std::vector<Eigen::Vector3f>* bindPositions = nullptr;
    const std::vector<Eigen::Vector3f>* bindNormals = nullptr;
    const std::vector<std::array<int, 4>>* boneIndices = nullptr;
    const std::vector<std::array<float, 4>>* boneWeights = nullptr;
    const std::vector<int>* boneNodeIndices = nullptr;
    const std::vector<Eigen::Matrix4f>* inverseBindMatrices = nullptr;
    const std::vector<Eigen::Matrix4f>* currentGlobalMatrices = nullptr;
};

struct CPUSkinningResult {
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> normals;
};

class Skinning {
  public:
    static std::vector<Eigen::Matrix4f> computeSkinningMatrices(
        const std::vector<int>& boneNodeIndices,
        const std::vector<Eigen::Matrix4f>& inverseBindMatrices,
        const std::vector<Eigen::Matrix4f>& currentGlobalMatrices);

    static void computeSkinningMatricesInto(
        const std::vector<int>& boneNodeIndices,
        const std::vector<Eigen::Matrix4f>& inverseBindMatrices,
        const std::vector<Eigen::Matrix4f>& currentGlobalMatrices,
        std::vector<Eigen::Matrix4f>& output);

    static std::vector<glm::mat4> computeSkinningMatrices(
        const std::vector<int>& boneNodeIndices,
        const std::vector<glm::mat4>& inverseBindMatrices,
        const std::vector<glm::mat4>& currentGlobalMatrices);

    static void computeSkinningMatricesInto(
        const std::vector<int>& boneNodeIndices,
        const std::vector<glm::mat4>& inverseBindMatrices,
        const std::vector<glm::mat4>& currentGlobalMatrices,
        std::vector<glm::mat4>& output);

    static CPUSkinningResult cpuSkin(const CPUSkinningInput& input);
};

} // namespace Animation
} // namespace KE

#endif
