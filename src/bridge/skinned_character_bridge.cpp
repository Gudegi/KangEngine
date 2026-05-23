#include "skinned_character_bridge.hpp"

#include "asset/fbx_loader.hpp"
#include "animation/skeleton_math.hpp"
#include "animation/skinning.hpp"
#include "engine/core/app/app.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/scene/native/prim.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <stdexcept>

namespace KE {
namespace Bridge {
namespace {

std::string primSafeName(std::string value, const std::string& fallback) {
    for (char& ch : value) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_'))
            ch = '_';
    }
    value.erase(std::remove(value.begin(), value.end(), '\0'), value.end());
    while (!value.empty() && value.front() == '_')
        value.erase(value.begin());
    while (!value.empty() && value.back() == '_')
        value.pop_back();
    return value.empty() ? fallback : value;
}

glm::vec4 diffuseColorFromMaterial(const Asset::FBXMeshMetadata& mesh,
                                   glm::vec4 fallback) {
    const int idx = mesh.primaryMaterialIndex;
    if (idx < 0 || idx >= static_cast<int>(mesh.materials.size()))
        return fallback;
    const auto& color = mesh.materials[static_cast<size_t>(idx)].diffuseColor;
    return glm::vec4(color[0], color[1], color[2], color[3]);
}

std::string
diffuseTexturePathFromMaterial(const Asset::FBXMeshMetadata& mesh) {
    const int idx = mesh.primaryMaterialIndex;
    if (idx < 0 || idx >= static_cast<int>(mesh.materials.size()))
        return {};
    const auto& material = mesh.materials[static_cast<size_t>(idx)];
    return material.hasDiffuseTexture ? material.diffuseTexturePath : "";
}

void validateBoneSlots(const SkinnedCharacterBridge::MeshBinding& mesh,
                       size_t motionNodeCount) {
    if (mesh.boneNames.size() != mesh.boneNodeIndices.size() ||
        mesh.boneNames.size() != mesh.inverseBindMatrices.size()) {
        throw std::runtime_error(
            "SkinnedCharacterBridge bone slot data size mismatch for mesh " +
            mesh.name);
    }

    for (size_t slot = 0; slot < mesh.boneNodeIndices.size(); ++slot) {
        const int node = mesh.boneNodeIndices[slot];
        if (node < 0 || node >= static_cast<int>(motionNodeCount)) {
            const std::string boneName =
                slot < mesh.boneNames.size() ? mesh.boneNames[slot]
                                             : std::to_string(slot);
            throw std::runtime_error(
                "SkinnedCharacterBridge bone '" + boneName +
                "' maps outside the current motion skeleton");
        }
    }
}

} // namespace

SkinnedCharacterBridge SkinnedCharacterBridge::fromFBX(
    App* app, Backend::Shader* shader, const std::string& fbxPath,
    const std::string& primBasePath, int clipIndex, float fps, float scale,
    bool useMaterials) {
    return fromFBXWithBind(app, shader, fbxPath, fbxPath, primBasePath,
                           clipIndex, fps, scale, useMaterials);
}

SkinnedCharacterBridge SkinnedCharacterBridge::fromFBXWithBind(
    App* app, Backend::Shader* shader, const std::string& motionFbxPath,
    const std::string& bindFbxPath, const std::string& primBasePath,
    int clipIndex, float fps, float scale, bool useMaterials) {
    if (!app)
        throw std::runtime_error("SkinnedCharacterBridge::fromFBX requires App");
    if (!shader)
        throw std::runtime_error(
            "SkinnedCharacterBridge::fromFBX requires Shader");

    SkinnedCharacterBridge bridge;
    bridge._app = app;
    Asset::FBXCharacterData character =
        (motionFbxPath == bindFbxPath)
            ? Asset::FBXLoader::loadCharacter(
                  motionFbxPath, clipIndex, fps, scale)
            : Asset::FBXLoader::loadCharacterWithBind(
                  motionFbxPath, bindFbxPath, clipIndex, fps, scale);
    bridge._motion = std::move(character.motion);
    std::vector<Asset::FBXSkinnedMeshInfo>& meshes =
        character.skinnedMeshes;
    bridge._meshes.reserve(meshes.size());

    static const glm::vec4 palette[] = {
        glm::vec4(0.82f, 0.78f, 0.68f, 1.0f),
        glm::vec4(0.42f, 0.58f, 0.92f, 1.0f),
        glm::vec4(0.88f, 0.45f, 0.34f, 1.0f),
        glm::vec4(0.55f, 0.78f, 0.48f, 1.0f),
    };

    const std::array<unsigned char, 4> whitePixel = {255, 255, 255, 255};
    Backend::TextureDesc whiteDesc;
    whiteDesc.width = 1;
    whiteDesc.height = 1;
    whiteDesc.channels = 4;
    whiteDesc.data = whitePixel.data();
    whiteDesc.name = "fbx_white_fallback";
    bridge._whiteTexture = app->getGraphicsDevice()->createTexture(whiteDesc);
    shader->use();
    shader->setInt("uTexture", 0);

    for (int i = 0; i < static_cast<int>(meshes.size()); ++i) {
        auto& imported = meshes[static_cast<size_t>(i)];
        const std::string safeName =
            primSafeName(imported.metadata.name, "mesh_" + std::to_string(i));
        const std::string path =
            primBasePath + "/" + std::to_string(i) + "_" + safeName;
        const glm::vec4 fallbackColor =
            palette[static_cast<size_t>(i) %
                    (sizeof(palette) / sizeof(palette[0]))];
        const glm::vec4 color =
            useMaterials ? diffuseColorFromMaterial(imported.metadata,
                                                    fallbackColor)
                         : fallbackColor;

        MeshBinding binding;
        binding.name = imported.metadata.name;
        binding.boneNames = imported.boneNames;
        binding.boneNodeIndices = imported.skinnedMeshData.boneNodeIndices;
        binding.inverseBindMatrices =
            imported.skinnedMeshData.inverseBindMatrices;
        validateBoneSlots(binding, bridge._motion.skeletonTree().numJoints());
        binding.boneMatrices.assign(binding.inverseBindMatrices.size(),
                                    glm::mat4(1.0f));
        binding.baseColor = color;
        binding.diffuseTexture = bridge._whiteTexture.get();

        App::MeshPrimResult result = app->addSkinnedMeshPrim(
            shader, path, std::move(imported.skinnedMeshData),
            glm::vec3(0.0f), color, true);
        binding.prim = result.prim;
        binding.handle = result.handle;

        const std::string texturePath =
            useMaterials ? diffuseTexturePathFromMaterial(imported.metadata)
                         : "";
        if (!texturePath.empty() && std::filesystem::exists(texturePath)) {
            bridge._textures.push_back(
                app->getGraphicsDevice()->createTexture(texturePath, true));
            binding.diffuseTexture = bridge._textures.back().get();
        }
        if (binding.handle != InvalidHandle && binding.diffuseTexture)
            app->setShapeTexture(binding.handle, binding.diffuseTexture, 0);

        bridge._meshes.push_back(std::move(binding));
    }

    bridge.applyTime(0.0f);
    return bridge;
}

Animation::SkeletonState SkinnedCharacterBridge::applyTime(float time,
                                                           bool loop) {
    const Animation::SkeletonState state = _motion.sample(time, loop);
    if (!_app)
        return state;

    state.computeGlobalTransformsInto(_globalTransforms);

    _globalMatrices.resize(_globalTransforms.size());
    for (size_t i = 0; i < _globalTransforms.size(); ++i)
        _globalMatrices[i] = Animation::transformToMat4(_globalTransforms[i]);

    for (MeshBinding& mesh : _meshes) {
        Animation::Skinning::computeSkinningMatricesInto(
            mesh.boneNodeIndices, mesh.inverseBindMatrices, _globalMatrices,
            mesh.boneMatrices);

        if (mesh.handle != InvalidHandle)
            _app->updateSkinningMatrices(mesh.handle, mesh.boneMatrices);
    }

    return state;
}

void SkinnedCharacterBridge::setVisible(bool visible) {
    const float alpha = visible ? 1.0f : 0.0f;
    for (MeshBinding& mesh : _meshes) {
        if (!mesh.prim)
            continue;
        glm::vec4 color = mesh.baseColor;
        color.a = alpha;
        mesh.prim->setDisplayColorAlpha(color);
    }
}

void SkinnedCharacterBridge::setColor(const glm::vec4& color) {
    for (MeshBinding& mesh : _meshes) {
        mesh.baseColor = color;
        if (mesh.prim)
            mesh.prim->setDisplayColorAlpha(color);
    }
}

void SkinnedCharacterBridge::setCastsShadow(bool castsShadow) {
    if (!_app)
        return;
    for (const MeshBinding& mesh : _meshes) {
        if (mesh.handle != InvalidHandle)
            _app->setShapeCastsShadow(mesh.handle, castsShadow);
    }
}

} // namespace Bridge
} // namespace KE
