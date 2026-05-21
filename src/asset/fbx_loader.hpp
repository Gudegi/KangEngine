#ifndef _FBX_LOADER_HPP_
#define _FBX_LOADER_HPP_

#include "animation/skeleton_tree.hpp"
#include "animation/skeleton_motion.hpp"
#include "engine/scene/scene_backend.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <string>
#include <vector>

namespace KE {
namespace Animation {

struct FBXAnimationClipInfo {
    std::string name;
    double startTime = 0.0;
    double endTime = 0.0;
    double frameRate = 0.0;
};

struct FBXMaterialInfo {
    std::string name;
    std::array<float, 4> diffuseColor = {1.0f, 1.0f, 1.0f, 1.0f};
    std::string diffuseTexturePath;
    bool hasDiffuseTexture = false;
    bool hasEmbeddedDiffuseTexture = false;
};

struct FBXMeshMetadata {
    std::string name;
    int vertexCount = 0;
    int indexCount = 0;
    int materialCount = 0;
    int primaryMaterialIndex = -1;
    bool hasNormals = false;
    bool hasUVs = false;
    bool hasSkin = false;
    std::vector<std::string> skinClusterNames;
    std::vector<FBXMaterialInfo> materials;
};

struct FBXMeshInfo : FBXMeshMetadata {
    Scene::MeshData meshData;
};

struct FBXSkinClusterInfo {
    std::string meshName;
    std::string clusterName;
    std::string linkName;
    int weightCount = 0;
    int indexCount = 0;
    int minIndex = -1;
    int maxIndex = -1;
    double minWeight = 0.0;
    double maxWeight = 0.0;
    double weightSum = 0.0;
    std::array<double, 3> transformTranslation = {0.0, 0.0, 0.0};
    std::array<double, 3> transformLinkTranslation = {0.0, 0.0, 0.0};
};

struct FBXSkinnedMeshInfo {
    FBXMeshMetadata mesh;
    Scene::SkinnedMeshData skinnedMeshData;
    std::vector<std::string> boneNames;
    std::vector<std::array<float, 16>> bindBoneMatrices;
    std::vector<std::array<float, 16>> bindMeshMatrices;
    int overweightVertexCount = 0;
    int unweightedVertexCount = 0;
    int mismatchedClusterCount = 0;
};

struct FBXCharacterAsset {
    SkeletonMotion motion;
    std::vector<FBXSkinnedMeshInfo> skinnedMeshes;
};

class FBXLoader {
  public:
    FBXLoader() = delete;

    // Phase-1 import: load FBX transform nodes into KangEngine's SkeletonTree.
    static SkeletonTree loadSkeleton(const std::string& fbxPath,
                                     float scale = 0.01f);

    static std::vector<FBXAnimationClipInfo>
    loadAnimationClipInfos(const std::string& fbxPath);

    static SkeletonMotion loadMotion(const std::string& fbxPath,
                                     int clipIndex = 0, float fps = 30.0f,
                                     float scale = 0.01f);

    static std::vector<FBXMeshInfo> loadMeshes(const std::string& fbxPath,
                                               float scale = 0.01f);

    static std::vector<FBXSkinClusterInfo>
    loadSkinClusterInfos(const std::string& fbxPath, float scale = 0.01f);

    static std::vector<FBXSkinnedMeshInfo>
    loadSkinnedMeshes(const std::string& fbxPath, float scale = 0.01f);

    static FBXCharacterAsset loadCharacter(const std::string& fbxPath,
                                           int clipIndex = 0,
                                           float fps = 30.0f,
                                           float scale = 0.01f);

    static FBXCharacterAsset loadCharacterWithBind(
        const std::string& motionFbxPath, const std::string& bindFbxPath,
        int clipIndex = 0, float fps = 30.0f, float scale = 0.01f);
};

} // namespace Animation
} // namespace KE

#endif
