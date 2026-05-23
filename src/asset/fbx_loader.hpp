#ifndef _FBX_LOADER_HPP_
#define _FBX_LOADER_HPP_

#include "animation/skeleton_tree.hpp"
#include "animation/skeleton_motion.hpp"
#include "asset/import_diagnostics.hpp"
#include "engine/scene/scene_backend.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <string>
#include <vector>

namespace KE {
namespace Asset {

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

struct FBXStaticMeshInfo {
    FBXMeshMetadata metadata;
    Scene::MeshData meshData;
};

struct FBXSkinnedMeshInfo {
    FBXMeshMetadata metadata;
    Scene::SkinnedMeshData skinnedMeshData;
    std::vector<std::string> boneNames;
    std::vector<std::array<float, 16>> bindBoneMatrices;
    std::vector<std::array<float, 16>> bindMeshMatrices;
    int overweightVertexCount = 0;
    int unweightedVertexCount = 0;
    int mismatchedClusterCount = 0;
};

struct FBXCharacterData {
    // Sampled animation clip converted to KangEngine skeleton motion.
    Animation::SkeletonMotion motion;
    // Skinned meshes with vertex data, weights, bind poses, and materials.
    std::vector<FBXSkinnedMeshInfo> skinnedMeshes;
};

struct FBXImportResult {
    // Runtime skinned character payload extracted from the source FBX.
    FBXCharacterData character;
    // Available animation clips/takes found in the source FBX.
    std::vector<FBXAnimationClipInfo> clips;
    // Non-fatal import warnings collected while parsing.
    ImportDiagnostics diagnostics;
};

class FBXLoader {
  public:
    FBXLoader() = delete;

    // Phase-1 import: load FBX transform nodes into KangEngine's SkeletonTree.
    static Animation::SkeletonTree loadSkeleton(const std::string& fbxPath,
                                                float scale = 0.01f);

    static std::vector<FBXAnimationClipInfo>
    loadAnimationClipInfos(const std::string& fbxPath);

    static Animation::SkeletonMotion
    loadMotion(const std::string& fbxPath, int clipIndex = 0,
               float fps = 30.0f, float scale = 0.01f);

    static std::vector<FBXStaticMeshInfo> loadMeshes(const std::string& fbxPath,
                                                     float scale = 0.01f);

    static std::vector<FBXSkinnedMeshInfo>
    loadSkinnedMeshes(const std::string& fbxPath, float scale = 0.01f);

    static FBXImportResult parse(const std::string& fbxPath, int clipIndex = 0,
                                 float fps = 30.0f, float scale = 0.01f);

    static FBXImportResult parseWithBind(const std::string& motionFbxPath,
                                         const std::string& bindFbxPath,
                                         int clipIndex = 0, float fps = 30.0f,
                                         float scale = 0.01f);

    static FBXCharacterData loadCharacter(const std::string& fbxPath,
                                         int clipIndex = 0, float fps = 30.0f,
                                         float scale = 0.01f);

    static FBXCharacterData loadCharacterWithBind(
        const std::string& motionFbxPath, const std::string& bindFbxPath,
        int clipIndex = 0, float fps = 30.0f, float scale = 0.01f);
};

namespace FBXDebug {

struct SkinClusterInfo {
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

std::vector<SkinClusterInfo> loadSkinClusterInfos(const std::string& fbxPath,
                                                  float scale = 0.01f);

} // namespace FBXDebug

} // namespace Asset
} // namespace KE

#endif
