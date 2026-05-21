#include "skinning.hpp"

#include <stdexcept>

namespace KE {
namespace Animation {

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

std::vector<Eigen::Matrix4f> Skinning::computeSkinningMatrices(
    const std::vector<int>& boneNodeIndices,
    const std::vector<Eigen::Matrix4f>& inverseBindMatrices,
    const std::vector<Eigen::Matrix4f>& currentGlobalMatrices) {
    std::vector<Eigen::Matrix4f> matrices;
    computeSkinningMatricesInto(boneNodeIndices, inverseBindMatrices,
                                currentGlobalMatrices, matrices);
    return matrices;
}

void Skinning::computeSkinningMatricesInto(
    const std::vector<int>& boneNodeIndices,
    const std::vector<Eigen::Matrix4f>& inverseBindMatrices,
    const std::vector<Eigen::Matrix4f>& currentGlobalMatrices,
    std::vector<Eigen::Matrix4f>& output) {
    output.assign(inverseBindMatrices.size(), Eigen::Matrix4f::Identity());

    for (size_t bone = 0; bone < inverseBindMatrices.size(); ++bone) {
        if (bone >= boneNodeIndices.size())
            continue;
        const int node = boneNodeIndices[bone];
        if (node < 0 || node >= static_cast<int>(currentGlobalMatrices.size()))
            continue;
        output[bone] = currentGlobalMatrices[static_cast<size_t>(node)] *
                       inverseBindMatrices[bone];
    }
}

std::vector<glm::mat4> Skinning::computeSkinningMatrices(
    const std::vector<int>& boneNodeIndices,
    const std::vector<glm::mat4>& inverseBindMatrices,
    const std::vector<glm::mat4>& currentGlobalMatrices) {
    std::vector<glm::mat4> matrices;
    computeSkinningMatricesInto(boneNodeIndices, inverseBindMatrices,
                                currentGlobalMatrices, matrices);
    return matrices;
}

void Skinning::computeSkinningMatricesInto(
    const std::vector<int>& boneNodeIndices,
    const std::vector<glm::mat4>& inverseBindMatrices,
    const std::vector<glm::mat4>& currentGlobalMatrices,
    std::vector<glm::mat4>& output) {
    output.assign(inverseBindMatrices.size(), glm::mat4(1.0f));

    for (size_t bone = 0; bone < inverseBindMatrices.size(); ++bone) {
        if (bone >= boneNodeIndices.size())
            continue;
        const int node = boneNodeIndices[bone];
        if (node < 0 || node >= static_cast<int>(currentGlobalMatrices.size()))
            continue;
        output[bone] = currentGlobalMatrices[static_cast<size_t>(node)] *
                       inverseBindMatrices[bone];
    }
}

CPUSkinningResult Skinning::cpuSkin(const CPUSkinningInput& input) {
    require(input.bindPositions, "Skinning::cpuSkin requires bind positions");
    require(input.bindNormals, "Skinning::cpuSkin requires bind normals");
    require(input.boneIndices, "Skinning::cpuSkin requires bone indices");
    require(input.boneWeights, "Skinning::cpuSkin requires bone weights");
    require(input.boneNodeIndices, "Skinning::cpuSkin requires bone node indices");
    require(input.inverseBindMatrices,
            "Skinning::cpuSkin requires inverse bind matrices");
    require(input.currentGlobalMatrices,
            "Skinning::cpuSkin requires current global matrices");

    const size_t vertexCount = input.bindPositions->size();
    require(input.bindNormals->size() == vertexCount,
            "Skinning::cpuSkin bind normals size mismatch");
    require(input.boneIndices->size() == vertexCount,
            "Skinning::cpuSkin bone indices size mismatch");
    require(input.boneWeights->size() == vertexCount,
            "Skinning::cpuSkin bone weights size mismatch");

    CPUSkinningResult result;
    result.positions.assign(vertexCount, Eigen::Vector3f::Zero());
    result.normals.assign(vertexCount, Eigen::Vector3f::Zero());

    for (size_t v = 0; v < vertexCount; ++v) {
        const Eigen::Vector4f bindPos((*input.bindPositions)[v].x(),
                                      (*input.bindPositions)[v].y(),
                                      (*input.bindPositions)[v].z(), 1.0f);
        const Eigen::Vector3f& bindNormal = (*input.bindNormals)[v];

        for (int slot = 0; slot < 4; ++slot) {
            const int bone = (*input.boneIndices)[v][slot];
            const float weight = (*input.boneWeights)[v][slot];
            if (bone < 0 || weight <= 1e-6f)
                continue;
            if (bone >= static_cast<int>(input.boneNodeIndices->size()) ||
                bone >= static_cast<int>(input.inverseBindMatrices->size()))
                continue;

            const int node = (*input.boneNodeIndices)[bone];
            if (node < 0 ||
                node >= static_cast<int>(input.currentGlobalMatrices->size()))
                continue;

            const Eigen::Matrix4f skinMatrix =
                (*input.currentGlobalMatrices)[static_cast<size_t>(node)] *
                (*input.inverseBindMatrices)[static_cast<size_t>(bone)];
            const Eigen::Vector4f skinnedPos = skinMatrix * bindPos;
            result.positions[v] += weight * skinnedPos.head<3>();
            result.normals[v] +=
                weight * (skinMatrix.block<3, 3>(0, 0) * bindNormal);
        }

        const float n = result.normals[v].norm();
        if (n > 1e-8f)
            result.normals[v] /= n;
    }

    return result;
}

} // namespace Animation
} // namespace KE
