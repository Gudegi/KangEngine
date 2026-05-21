#ifndef _SKINNED_CHARACTER_BRIDGE_HPP_
#define _SKINNED_CHARACTER_BRIDGE_HPP_

#include "animation/skeleton_motion.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/scene/scene_backend.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <memory>
#include <string>
#include <vector>

namespace KE {

class App;

namespace Backend {
class Shader;
class Texture;
}

namespace Bridge {

// Connects imported skinned character assets to App rendering.
//
// Asset/FBXLoader owns import and conversion. This bridge owns the scene prims,
// mesh handles, and per-frame bone matrix upload for animation playback.
class SkinnedCharacterBridge {
  public:
    struct MeshBinding {
        std::string name;
        Scene::Prim* prim = nullptr;
        MeshHandle handle = InvalidHandle;
        std::vector<std::string> boneNames;
        std::vector<int> boneNodeIndices;
        std::vector<glm::mat4> inverseBindMatrices;
        std::vector<glm::mat4> boneMatrices;
        glm::vec4 baseColor = glm::vec4(1.0f);
        Backend::Texture* diffuseTexture = nullptr;
    };

    static SkinnedCharacterBridge fromFBX(
        App* app, Backend::Shader* shader, const std::string& fbxPath,
        const std::string& primBasePath = "/fbx_character",
        int clipIndex = 0, float fps = 30.0f, float scale = 0.01f,
        bool useMaterials = true);

    static SkinnedCharacterBridge fromFBXWithBind(
        App* app, Backend::Shader* shader, const std::string& motionFbxPath,
        const std::string& bindFbxPath,
        const std::string& primBasePath = "/fbx_character",
        int clipIndex = 0, float fps = 30.0f, float scale = 0.01f,
        bool useMaterials = true);

    Animation::SkeletonState applyTime(float time, bool loop = true);
    void setVisible(bool visible);
    void setColor(const glm::vec4& color);
    void setCastsShadow(bool castsShadow);

    const Animation::SkeletonMotion& motion() const { return _motion; }
    const std::vector<MeshBinding>& meshes() const { return _meshes; }
    std::vector<MeshBinding>& meshes() { return _meshes; }

  private:
    App* _app = nullptr;
    Animation::SkeletonMotion _motion;
    std::vector<MeshBinding> _meshes;
    std::vector<Animation::Transform> _globalTransforms;
    std::vector<glm::mat4> _globalMatrices;
    std::vector<std::unique_ptr<Backend::Texture>> _textures;
    std::unique_ptr<Backend::Texture> _whiteTexture;
};

} // namespace Bridge
} // namespace KE

#endif
