#include "asset/fbx_loader.hpp"

#include <fmt/core.h>
#include <ofbx.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace KE {
namespace Animation {

namespace {

struct FBXSkeletonPoseSample {
    std::vector<std::string> nodeNames;
    std::vector<int> parentIndices;
    std::vector<Eigen::Vector3f> localTranslations;
    std::vector<Eigen::Quaternionf> localRotations;
    Eigen::Vector3f rootTranslation = Eigen::Vector3f::Zero();
};

const char* upAxisName(ofbx::UpVector axis) {
    switch (axis) {
    case ofbx::UpVector_AxisX:
        return "X";
    case ofbx::UpVector_AxisY:
        return "Y";
    case ofbx::UpVector_AxisZ:
        return "Z";
    default:
        return "unknown";
    }
}

const char* coordSystemName(ofbx::CoordSystem coord) {
    switch (coord) {
    case ofbx::CoordSystem_RightHanded:
        return "right-handed";
    case ofbx::CoordSystem_LeftHanded:
        return "left-handed";
    default:
        return "unknown";
    }
}

void warnUnexpectedGlobalSettings(const ofbx::IScene& scene,
                                  const std::string& fbxPath) {
    const ofbx::GlobalSettings* settings = scene.getGlobalSettings();
    if (!settings)
        return;

    const bool yUp = settings->UpAxis == ofbx::UpVector_AxisY &&
                     settings->UpAxisSign == 1;
    const bool rightHanded =
        settings->CoordAxis == ofbx::CoordSystem_RightHanded &&
        settings->CoordAxisSign == 1;
    if (yUp && rightHanded)
        return;

    fmt::print(stderr,
               "FBX warning: '{}' uses up={} sign={} coord={} sign={}; "
               "KangEngine FBX import currently assumes Y-up, right-handed "
               "data and does not apply automatic axis conversion.\n",
               fbxPath, upAxisName(settings->UpAxis), settings->UpAxisSign,
               coordSystemName(settings->CoordAxis), settings->CoordAxisSign);
}

std::vector<ofbx::u8> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error(fmt::format("Failed to open FBX file: {}", path));

    const std::streamsize size = file.tellg();
    if (size <= 0)
        throw std::runtime_error(fmt::format("FBX file is empty: {}", path));

    std::vector<ofbx::u8> bytes(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error(fmt::format("Failed to read FBX file: {}", path));
    }
    return bytes;
}

ofbx::IScene* loadSceneOrThrow(const std::string& fbxPath, ofbx::u16 flags) {
    const auto bytes = readBinaryFile(fbxPath);
    ofbx::IScene* scene = ofbx::load(bytes.data(), bytes.size(), flags);
    if (!scene) {
        const char* error = ofbx::getError();
        throw std::runtime_error(fmt::format("Failed to load FBX file '{}': {}",
                                             fbxPath,
                                             error ? error : "unknown error"));
    }
    warnUnexpectedGlobalSettings(*scene, fbxPath);
    return scene;
}

bool isSkeletonNode(const ofbx::Object& object) {
    return object.isNode() &&
           (object.getType() == ofbx::Object::Type::LIMB_NODE ||
            object.getType() == ofbx::Object::Type::NULL_NODE);
}

bool hasSkeletonNodes(const ofbx::IScene& scene) {
    const ofbx::Object* const* objects = scene.getAllObjects();
    for (int i = 0; i < scene.getAllObjectCount(); ++i) {
        if (objects[i] && isSkeletonNode(*objects[i]))
            return true;
    }
    return false;
}

bool isFallbackTransformNode(const ofbx::Object& object) {
    if (!object.isNode())
        return false;

    switch (object.getType()) {
    case ofbx::Object::Type::MESH:
    case ofbx::Object::Type::GEOMETRY:
    case ofbx::Object::Type::MATERIAL:
    case ofbx::Object::Type::TEXTURE:
    case ofbx::Object::Type::CAMERA:
    case ofbx::Object::Type::LIGHT:
    case ofbx::Object::Type::ANIMATION_STACK:
    case ofbx::Object::Type::ANIMATION_LAYER:
    case ofbx::Object::Type::ANIMATION_CURVE:
    case ofbx::Object::Type::ANIMATION_CURVE_NODE:
        return false;
    default:
        return object.getType() != ofbx::Object::Type::ROOT;
    }
}

std::string makeNodeName(const ofbx::Object& object, int index) {
    std::string name = object.name;
    if (name.empty())
        name = fmt::format("fbx_node_{}", index);
    return name;
}

std::string dataViewToString(ofbx::DataView view) {
    if (!view.begin || !view.end || view.end <= view.begin)
        return {};
    std::string result;
    result.reserve(static_cast<size_t>(view.end - view.begin));
    for (const ofbx::u8* it = view.begin; it != view.end; ++it)
        result.push_back(static_cast<char>(*it));
    return result;
}

Eigen::Quaternionf rotationFromMatrix(const ofbx::DMatrix& transform) {
    // OpenFBX matrices store translation in the last four values. The upper 3x3
    // may include scale, so normalize the basis before extracting rotation.
    Eigen::Vector3f x(static_cast<float>(transform.m[0]),
                      static_cast<float>(transform.m[1]),
                      static_cast<float>(transform.m[2]));
    Eigen::Vector3f y(static_cast<float>(transform.m[4]),
                      static_cast<float>(transform.m[5]),
                      static_cast<float>(transform.m[6]));
    Eigen::Vector3f z(static_cast<float>(transform.m[8]),
                      static_cast<float>(transform.m[9]),
                      static_cast<float>(transform.m[10]));

    if (x.norm() <= 1e-6f || y.norm() <= 1e-6f || z.norm() <= 1e-6f)
        return Eigen::Quaternionf::Identity();

    x.normalize();
    y.normalize();
    z.normalize();

    Eigen::Matrix3f rotation;
    rotation.col(0) = x;
    rotation.col(1) = y;
    rotation.col(2) = z;

    Eigen::Quaternionf q(rotation);
    if (q.norm() <= 1e-6f)
        return Eigen::Quaternionf::Identity();
    return q.normalized();
}

Eigen::Vector3f translationFromMatrix(const ofbx::DMatrix& transform,
                                      float scale) {
    return Eigen::Vector3f(static_cast<float>(transform.m[12]) * scale,
                           static_cast<float>(transform.m[13]) * scale,
                           static_cast<float>(transform.m[14]) * scale);
}

std::vector<const ofbx::Object*> orderedImportNodes(const ofbx::IScene& scene) {
    const bool skeletonOnly = hasSkeletonNodes(scene);
    const ofbx::Object* const* objects = scene.getAllObjects();

    std::vector<const ofbx::Object*> selected;
    selected.reserve(static_cast<size_t>(scene.getAllObjectCount()));

    for (int i = 0; i < scene.getAllObjectCount(); ++i) {
        const ofbx::Object* object = objects[i];
        if (!object)
            continue;

        const bool useNode =
            skeletonOnly ? isSkeletonNode(*object)
                         : isFallbackTransformNode(*object);
        if (useNode)
            selected.push_back(object);
    }

    if (selected.empty())
        return selected;

    std::unordered_map<const ofbx::Object*, bool> selectedSet;
    selectedSet.reserve(selected.size());
    for (const ofbx::Object* object : selected)
        selectedSet[object] = true;

    std::unordered_map<const ofbx::Object*, std::vector<const ofbx::Object*>>
        children;
    std::vector<const ofbx::Object*> roots;
    children.reserve(selected.size());

    auto nearestSelectedParent = [&](const ofbx::Object* object) {
        for (const ofbx::Object* parent = object->getParent(); parent;
             parent = parent->getParent()) {
            if (selectedSet.find(parent) != selectedSet.end())
                return parent;
        }
        return static_cast<const ofbx::Object*>(nullptr);
    };

    for (const ofbx::Object* object : selected) {
        const ofbx::Object* parent = nearestSelectedParent(object);
        if (parent)
            children[parent].push_back(object);
        else
            roots.push_back(object);
    }

    std::vector<const ofbx::Object*> ordered;
    ordered.reserve(selected.size());
    std::function<void(const ofbx::Object*)> appendTree =
        [&](const ofbx::Object* object) {
            ordered.push_back(object);
            auto childIt = children.find(object);
            if (childIt == children.end())
                return;
            for (const ofbx::Object* child : childIt->second)
                appendTree(child);
        };

    for (const ofbx::Object* root : roots)
        appendTree(root);

    return ordered;
}

std::vector<int> parentIndicesForOrderedNodes(
    const std::vector<const ofbx::Object*>& ordered) {
    std::unordered_map<const ofbx::Object*, int> objectToIndex;
    objectToIndex.reserve(ordered.size());
    for (int i = 0; i < static_cast<int>(ordered.size()); ++i)
        objectToIndex[ordered[i]] = i;

    std::vector<int> parentIndices;
    parentIndices.reserve(ordered.size());
    for (const ofbx::Object* object : ordered) {
        int parentIndex = -1;
        for (const ofbx::Object* parent = object->getParent(); parent;
             parent = parent->getParent()) {
            auto it = objectToIndex.find(parent);
            if (it != objectToIndex.end()) {
                parentIndex = it->second;
                break;
            }
        }
        parentIndices.push_back(parentIndex);
    }
    return parentIndices;
}

std::vector<FBXAnimationClipInfo> clipInfosFromScene(const ofbx::IScene& scene) {
    std::vector<FBXAnimationClipInfo> clips;
    clips.reserve(static_cast<size_t>(scene.getAnimationStackCount()));

    const double sceneFrameRate = scene.getSceneFrameRate();
    for (int i = 0; i < scene.getAnimationStackCount(); ++i) {
        const ofbx::AnimationStack* stack = scene.getAnimationStack(i);
        if (!stack)
            continue;

        FBXAnimationClipInfo clip;
        clip.name = stack->name;
        clip.frameRate = sceneFrameRate;

        const ofbx::TakeInfo* takeInfo = scene.getTakeInfo(stack->name);
        if (takeInfo) {
            clip.name = dataViewToString(takeInfo->name);
            clip.startTime = takeInfo->local_time_from;
            clip.endTime = takeInfo->local_time_to;
        }

        clips.push_back(std::move(clip));
    }

    return clips;
}

FBXSkeletonPoseSample sampleSkeletonPoseFromScene(
    const ofbx::IScene& scene, const std::vector<const ofbx::Object*>& ordered,
    double time, int clipIndex, float scale, const std::string& sourceName) {
    if (ordered.empty()) {
        throw std::runtime_error(fmt::format(
            "FBX file '{}' has no importable transform nodes", sourceName));
    }

    if (clipIndex < 0 || clipIndex >= scene.getAnimationStackCount()) {
        throw std::runtime_error(fmt::format(
            "FBX clip index {} is out of range [0, {})", clipIndex,
            scene.getAnimationStackCount()));
    }

    const ofbx::AnimationStack* stack = scene.getAnimationStack(clipIndex);
    const ofbx::AnimationLayer* layer = stack ? stack->getLayer(0) : nullptr;

    FBXSkeletonPoseSample sample;
    sample.parentIndices = parentIndicesForOrderedNodes(ordered);
    sample.nodeNames.reserve(ordered.size());
    sample.localTranslations.reserve(ordered.size());
    sample.localRotations.reserve(ordered.size());

    for (int i = 0; i < static_cast<int>(ordered.size()); ++i) {
        const ofbx::Object* object = ordered[i];
        ofbx::DVec3 translation = object->getLocalTranslation();
        ofbx::DVec3 rotation = object->getLocalRotation();

        if (layer) {
            if (const ofbx::AnimationCurveNode* translationCurve =
                    layer->getCurveNode(*object, "Lcl Translation")) {
                translation = translationCurve->getNodeLocalTransform(time);
            }
            if (const ofbx::AnimationCurveNode* rotationCurve =
                    layer->getCurveNode(*object, "Lcl Rotation")) {
                rotation = rotationCurve->getNodeLocalTransform(time);
            }
        }

        const ofbx::DMatrix localTransform =
            object->evalLocal(translation, rotation, object->getLocalScaling());
        sample.nodeNames.push_back(makeNodeName(*object, i));
        sample.localTranslations.push_back(
            translationFromMatrix(localTransform, scale));
        sample.localRotations.push_back(rotationFromMatrix(localTransform));
    }

    if (!sample.localTranslations.empty())
        sample.rootTranslation = sample.localTranslations.front();

    return sample;
}

glm::vec3 glmVec3FromFbx(const ofbx::Vec3& v, float scale) {
    return glm::vec3(static_cast<float>(v.x) * scale,
                     static_cast<float>(v.y) * scale,
                     static_cast<float>(v.z) * scale);
}

glm::vec3 glmNormalFromFbx(const ofbx::Vec3& v) {
    return glm::vec3(static_cast<float>(v.x), static_cast<float>(v.y),
                     static_cast<float>(v.z));
}

glm::vec2 glmVec2FromFbx(const ofbx::Vec2& v) {
    return glm::vec2(static_cast<float>(v.x), static_cast<float>(v.y));
}

const ofbx::Skin* skinForMesh(const ofbx::Mesh& mesh) {
    const ofbx::Skin* skin = mesh.getSkin();
    if (!skin && mesh.getGeometry())
        skin = mesh.getGeometry()->getSkin();
    return skin;
}

std::string normalizeTexturePathString(std::string path) {
    std::replace(path.begin(), path.end(), '\\',
                 std::filesystem::path::preferred_separator);
    return path;
}

std::string resolveTexturePath(const std::string& fbxPath,
                               const ofbx::Texture& texture) {
    std::string relative =
        normalizeTexturePathString(dataViewToString(texture.getRelativeFileName()));
    std::string file =
        normalizeTexturePathString(dataViewToString(texture.getFileName()));

    std::filesystem::path fbxDir =
        std::filesystem::path(fbxPath).parent_path();
    auto resolveCandidate = [&](const std::string& value) -> std::string {
        if (value.empty())
            return {};
        std::filesystem::path path(value);
        if (path.is_absolute())
            return path.string();
        std::filesystem::path joined = fbxDir / path;
        if (std::filesystem::exists(joined))
            return joined.string();
        return joined.lexically_normal().string();
    };

    std::string resolved = resolveCandidate(relative);
    if (!resolved.empty())
        return resolved;
    return resolveCandidate(file);
}

FBXMaterialInfo materialInfoFromMaterial(const std::string& fbxPath,
                                         const ofbx::Material& material) {
    FBXMaterialInfo info;
    info.name = material.name;

    const ofbx::Color diffuse = material.getDiffuseColor();
    const float factor = static_cast<float>(material.getDiffuseFactor());
    info.diffuseColor = {diffuse.r * factor, diffuse.g * factor,
                         diffuse.b * factor, 1.0f};

    if (const ofbx::Texture* texture =
            material.getTexture(ofbx::Texture::DIFFUSE)) {
        info.diffuseTexturePath = resolveTexturePath(fbxPath, *texture);
        info.hasDiffuseTexture = !info.diffuseTexturePath.empty();
        info.hasEmbeddedDiffuseTexture =
            texture->getEmbeddedData().begin != texture->getEmbeddedData().end;
    }

    return info;
}

std::vector<FBXMaterialInfo> materialInfosForMesh(const std::string& fbxPath,
                                                  const ofbx::Mesh& mesh) {
    std::vector<FBXMaterialInfo> materials;
    materials.reserve(static_cast<size_t>(mesh.getMaterialCount()));
    for (int materialIdx = 0; materialIdx < mesh.getMaterialCount();
         ++materialIdx) {
        const ofbx::Material* material = mesh.getMaterial(materialIdx);
        if (!material)
            continue;
        materials.push_back(materialInfoFromMaterial(fbxPath, *material));
    }
    return materials;
}

int findNodeIndexByName(const std::vector<std::string>& names,
                        const std::string& target) {
    auto it = std::find(names.begin(), names.end(), target);
    if (it == names.end())
        return -1;
    return static_cast<int>(std::distance(names.begin(), it));
}

Eigen::Matrix4f eigenMatrixFromFbx(const ofbx::DMatrix& transform,
                                   float scale) {
    Eigen::Matrix4f out;
    // OpenFBX stores DMatrix in column-major order. Eigen uses column-major by
    // default too, but assign explicitly so the convention is visible here.
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            out(row, col) =
                static_cast<float>(transform.m[row + col * 4]);
        }
    }
    out(0, 3) *= scale;
    out(1, 3) *= scale;
    out(2, 3) *= scale;
    return out;
}

std::array<float, 16> rowMajorArrayFromEigen(const Eigen::Matrix4f& m) {
    std::array<float, 16> out{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col)
            out[static_cast<size_t>(row * 4 + col)] = m(row, col);
    }
    return out;
}

glm::mat4 glmMat4FromRowMajorArray(const std::array<float, 16>& a) {
    glm::mat4 out(1.0f);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col)
            out[col][row] = a[static_cast<size_t>(row * 4 + col)];
    }
    return out;
}

void addVertexInfluence(glm::ivec4& indices, glm::vec4& weights,
                        int boneIndex, float weight,
                        bool& droppedInfluence) {
    if (weight <= 0.0f)
        return;

    for (int i = 0; i < 4; ++i) {
        if (weights[i] == 0.0f) {
            indices[i] = boneIndex;
            weights[i] = weight;
            return;
        }
    }

    int minSlot = 0;
    for (int i = 1; i < 4; ++i) {
        if (weights[i] < weights[minSlot])
            minSlot = i;
    }
    if (weight > weights[minSlot]) {
        indices[minSlot] = boneIndex;
        weights[minSlot] = weight;
    }
    droppedInfluence = true;
}

void normalizeWeights(glm::vec4& weights) {
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i)
        sum += weights[i];
    if (sum <= 1e-8f)
        return;
    for (int i = 0; i < 4; ++i)
        weights[i] /= sum;
}

uint32_t floatBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    return bits;
}

struct MeshVertexKey {
    std::array<uint32_t, 3> position{};
    std::array<uint32_t, 3> normal{};
    std::array<uint32_t, 2> uv{};
    int controlPoint = -1;
    unsigned int uniqueCorner = 0;

    bool operator==(const MeshVertexKey& other) const {
        return position == other.position && normal == other.normal &&
               uv == other.uv && controlPoint == other.controlPoint &&
               uniqueCorner == other.uniqueCorner;
    }
};

struct MeshVertexKeyHash {
    size_t operator()(const MeshVertexKey& key) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) {
            h ^= static_cast<size_t>(v);
            h *= 1099511628211ull;
        };
        for (uint32_t v : key.position)
            mix(v);
        for (uint32_t v : key.normal)
            mix(v);
        for (uint32_t v : key.uv)
            mix(v);
        mix(static_cast<uint32_t>(key.controlPoint));
        mix(key.uniqueCorner);
        return h;
    }
};

MeshVertexKey makeMeshVertexKey(const glm::vec3& position,
                                const glm::vec3& normal, const glm::vec2& uv,
                                int controlPoint, unsigned int uniqueCorner) {
    MeshVertexKey key;
    key.position = {floatBits(position.x), floatBits(position.y),
                    floatBits(position.z)};
    key.normal = {floatBits(normal.x), floatBits(normal.y),
                  floatBits(normal.z)};
    key.uv = {floatBits(uv.x), floatBits(uv.y)};
    key.controlPoint = controlPoint;
    key.uniqueCorner = uniqueCorner;
    return key;
}

void triangulateMeshGeometry(const std::string& meshName,
                             const ofbx::GeometryData& geom, float scale,
                             Scene::MeshData& meshData,
                             std::vector<int>* renderToControlPoint = nullptr) {
    const ofbx::Vec3Attributes positions = geom.getPositions();
    const ofbx::Vec3Attributes normals = geom.getNormals();
    const ofbx::Vec2Attributes uvs = geom.getUVs();
    const bool hasNormals = normals.values != nullptr;
    const bool hasUVs = uvs.values != nullptr;

    std::unordered_map<MeshVertexKey, unsigned int, MeshVertexKeyHash>
        dedupedVertices;
    dedupedVertices.reserve(static_cast<size_t>(std::max(0, positions.count)));

    constexpr int TriBufferSize = 256;
    constexpr int MaxPolygonVertices = TriBufferSize / 3;
    int triBuffer[TriBufferSize] = {};
    for (int partitionIdx = 0; partitionIdx < geom.getPartitionCount();
         ++partitionIdx) {
        const ofbx::GeometryPartition partition =
            geom.getPartition(partitionIdx);
        for (int polygonIdx = 0; polygonIdx < partition.polygon_count;
             ++polygonIdx) {
            const ofbx::GeometryPartition::Polygon& polygon =
                partition.polygons[polygonIdx];
            if (polygon.vertex_count < 3)
                continue;
            if (polygon.vertex_count > MaxPolygonVertices)
                throw std::runtime_error(fmt::format(
                    "FBX mesh '{}' has polygon with {} vertices; "
                    "increase triangulation buffer above {}",
                    meshName, polygon.vertex_count, TriBufferSize));

            const ofbx::u32 triCount =
                ofbx::triangulate(geom, polygon, triBuffer);
            for (ofbx::u32 i = 0; i < triCount; ++i) {
                const int srcIndex = triBuffer[i];
                const int controlPoint =
                    positions.indices ? positions.indices[srcIndex] : srcIndex;
                const glm::vec3 position =
                    glmVec3FromFbx(positions.get(srcIndex), scale);
                const glm::vec3 normal =
                    hasNormals ? glmNormalFromFbx(normals.get(srcIndex))
                               : glm::vec3(0.0f);
                const glm::vec2 uv =
                    hasUVs ? glmVec2FromFbx(uvs.get(srcIndex))
                           : glm::vec2(0.0f);

                // If normals are missing, keep one vertex per face corner so
                // fillMissingAttributes() can preserve flat-shaded normals.
                const unsigned int uniqueCorner =
                    hasNormals
                        ? 0u
                        : static_cast<unsigned int>(meshData.indices.size());
                const MeshVertexKey key = makeMeshVertexKey(
                    position, normal, uv, controlPoint, uniqueCorner);

                auto [it, inserted] = dedupedVertices.emplace(
                    key, static_cast<unsigned int>(meshData.vertices.size()));
                const unsigned int dstIndex = it->second;
                if (inserted) {
                    meshData.vertices.push_back(position);
                    if (hasNormals)
                        meshData.normals.push_back(normal);
                    if (hasUVs)
                        meshData.uvs.push_back(uv);
                    if (renderToControlPoint)
                        renderToControlPoint->push_back(controlPoint);
                }
                meshData.indices.push_back(dstIndex);
            }
        }
    }

    meshData.fillMissingAttributes();
}

SkeletonMotion loadMotionFromScene(const std::string& fbxPath,
                                   const ofbx::IScene& scene, int clipIndex,
                                   float fps, float scale) {
    if (fps <= 0.0f)
        throw std::runtime_error("FBX motion fps must be positive");

    const std::vector<FBXAnimationClipInfo> clips = clipInfosFromScene(scene);
    if (clipIndex < 0 || clipIndex >= static_cast<int>(clips.size())) {
        throw std::runtime_error(fmt::format(
            "FBX clip index {} is out of range [0, {})", clipIndex,
            static_cast<int>(clips.size())));
    }

    const FBXAnimationClipInfo& clip = clips[clipIndex];
    const double start = clip.startTime;
    const double end = clip.endTime;
    if (end < start)
        throw std::runtime_error("FBX clip end time is before start time");

    const int frameCount =
        std::max(1, static_cast<int>(std::floor((end - start) * fps)) + 1);

    const std::vector<const ofbx::Object*> ordered = orderedImportNodes(scene);
    FBXSkeletonPoseSample firstSample = sampleSkeletonPoseFromScene(
        scene, ordered, start, clipIndex, scale, fbxPath);
    const int numJoints = static_cast<int>(firstSample.nodeNames.size());

    std::vector<Eigen::Quaternionf> restRotations;
    restRotations.reserve(static_cast<size_t>(numJoints));
    for (const Eigen::Quaternionf& q : firstSample.localRotations)
        restRotations.push_back(q);

    std::shared_ptr<SkeletonTree> tree = std::make_shared<SkeletonTree>(
        firstSample.nodeNames, firstSample.parentIndices,
        firstSample.localTranslations, restRotations,
        std::vector<int>(static_cast<size_t>(numJoints), 0));

    std::vector<float> rootTranslations;
    rootTranslations.reserve(static_cast<size_t>(frameCount) * 3);
    std::vector<float> localRotationsWxyz;
    localRotationsWxyz.reserve(static_cast<size_t>(frameCount) *
                               static_cast<size_t>(numJoints) * 4);

    for (int f = 0; f < frameCount; ++f) {
        const double time =
            start + static_cast<double>(f) / static_cast<double>(fps);
        FBXSkeletonPoseSample sample =
            (f == 0) ? std::move(firstSample)
                     : sampleSkeletonPoseFromScene(
                           scene, ordered, time, clipIndex, scale, fbxPath);

        rootTranslations.push_back(sample.rootTranslation.x());
        rootTranslations.push_back(sample.rootTranslation.y());
        rootTranslations.push_back(sample.rootTranslation.z());

        for (const Eigen::Quaternionf& qRaw : sample.localRotations) {
            Eigen::Quaternionf q = qRaw;
            if (q.norm() <= 1e-6f)
                q = Eigen::Quaternionf::Identity();
            else
                q.normalize();
            localRotationsWxyz.push_back(q.w());
            localRotationsWxyz.push_back(q.x());
            localRotationsWxyz.push_back(q.y());
            localRotationsWxyz.push_back(q.z());
        }
    }

    return SkeletonMotion(std::move(tree), fps, clip.name,
                          std::move(rootTranslations),
                          std::move(localRotationsWxyz));
}

void remapSkinnedMeshesToMotionByName(
    std::vector<FBXSkinnedMeshInfo>& meshes, const SkeletonMotion& motion) {
    std::unordered_map<std::string, int> sourceIndexByName;
    const std::vector<std::string>& sourceNames =
        motion.skeletonTree().nodeNames();
    sourceIndexByName.reserve(sourceNames.size());
    for (int i = 0; i < static_cast<int>(sourceNames.size()); ++i)
        sourceIndexByName[sourceNames[static_cast<size_t>(i)]] = i;

    for (FBXSkinnedMeshInfo& mesh : meshes) {
        mesh.skinnedMeshData.boneNodeIndices.clear();
        mesh.skinnedMeshData.boneNodeIndices.reserve(mesh.boneNames.size());
        for (const std::string& boneName : mesh.boneNames) {
            auto it = sourceIndexByName.find(boneName);
            if (it == sourceIndexByName.end()) {
                throw std::runtime_error(fmt::format(
                    "FBX skin bone '{}' is missing from motion skeleton",
                    boneName));
            }
            mesh.skinnedMeshData.boneNodeIndices.push_back(it->second);
        }
    }
}

void transformMeshData(Scene::MeshData& meshData, const glm::mat4& transform) {
    const glm::mat3 normalTransform =
        glm::transpose(glm::inverse(glm::mat3(transform)));

    for (glm::vec3& position : meshData.vertices) {
        const glm::vec4 p = transform * glm::vec4(position, 1.0f);
        position = glm::vec3(p);
    }

    for (glm::vec3& normal : meshData.normals) {
        const glm::vec3 n = normalTransform * normal;
        const float len = glm::length(n);
        normal = (len > 1e-8f) ? (n / len) : glm::vec3(0.0f, 1.0f, 0.0f);
    }
}

void bakeBindMeshTransforms(std::vector<FBXSkinnedMeshInfo>& meshes) {
    for (FBXSkinnedMeshInfo& mesh : meshes) {
        if (mesh.bindMeshMatrices.empty() || mesh.bindBoneMatrices.empty())
            continue;
        // Split bind/motion assets such as the Geno FBX set store skinned
        // vertices below the skeleton bind space. Bake the root bind bone and
        // mesh bind transforms once so runtime skin matrices can stay
        // currentBoneGlobal * inverseBindBone.
        const glm::mat4 rootBind =
            glmMat4FromRowMajorArray(mesh.bindBoneMatrices[0]);
        const glm::mat4 meshBind =
            glmMat4FromRowMajorArray(mesh.bindMeshMatrices[0]);
        transformMeshData(mesh.skinnedMeshData.mesh, rootBind * meshBind);
    }
}

std::vector<FBXSkinnedMeshInfo>
loadSkinnedMeshesFromScene(const std::string& fbxPath,
                           const ofbx::IScene& scene, float scale) {
    const std::vector<const ofbx::Object*> orderedNodes =
        orderedImportNodes(scene);
    std::vector<std::string> nodeNames;
    nodeNames.reserve(orderedNodes.size());
    for (int i = 0; i < static_cast<int>(orderedNodes.size()); ++i)
        nodeNames.push_back(makeNodeName(*orderedNodes[i], i));

    std::vector<FBXSkinnedMeshInfo> result;
    result.reserve(static_cast<size_t>(scene.getMeshCount()));

    for (int meshIdx = 0; meshIdx < scene.getMeshCount(); ++meshIdx) {
        const ofbx::Mesh* mesh = scene.getMesh(meshIdx);
        if (!mesh)
            continue;

        const ofbx::GeometryData& geom = mesh->getGeometryData();
        const ofbx::Vec3Attributes positions = geom.getPositions();
        const ofbx::Vec3Attributes normals = geom.getNormals();
        const ofbx::Vec2Attributes uvs = geom.getUVs();

        FBXSkinnedMeshInfo info;
        info.mesh.name = makeNodeName(*mesh, meshIdx);
        info.mesh.materialCount = mesh->getMaterialCount();
        info.mesh.materials = materialInfosForMesh(fbxPath, *mesh);
        if (!info.mesh.materials.empty())
            info.mesh.primaryMaterialIndex = 0;
        info.mesh.hasNormals = normals.values != nullptr;
        info.mesh.hasUVs = uvs.values != nullptr;

        std::vector<int> renderToControlPoint;
        renderToControlPoint.reserve(
            static_cast<size_t>(std::max(0, positions.count)));

        triangulateMeshGeometry(info.mesh.name, geom, scale,
                                info.skinnedMeshData.mesh,
                                &renderToControlPoint);
        info.mesh.vertexCount =
            static_cast<int>(info.skinnedMeshData.mesh.vertices.size());
        info.mesh.indexCount =
            static_cast<int>(info.skinnedMeshData.mesh.indices.size());

        info.skinnedMeshData.boneIndices.assign(
            info.skinnedMeshData.mesh.vertices.size(), glm::ivec4(-1));
        info.skinnedMeshData.boneWeights.assign(
            info.skinnedMeshData.mesh.vertices.size(), glm::vec4(0.0f));

        const ofbx::Skin* skin = skinForMesh(*mesh);
        info.mesh.hasSkin = skin != nullptr;
        if (skin) {
            info.mesh.skinClusterNames.reserve(
                static_cast<size_t>(skin->getClusterCount()));
            std::unordered_map<int, std::vector<std::pair<int, float>>>
                controlPointInfluences;

            for (int clusterIdx = 0; clusterIdx < skin->getClusterCount();
                 ++clusterIdx) {
                const ofbx::Cluster* cluster = skin->getCluster(clusterIdx);
                if (!cluster)
                    continue;
                const ofbx::Object* link = cluster->getLink();
                const std::string boneName =
                    link ? std::string(link->name)
                         : fmt::format("cluster_link_{}", clusterIdx);
                info.mesh.skinClusterNames.push_back(boneName);

                const int boneSlot = static_cast<int>(info.boneNames.size());
                info.boneNames.push_back(boneName);
                info.skinnedMeshData.boneNodeIndices.push_back(
                    findNodeIndexByName(nodeNames, boneName));
                const Eigen::Matrix4f bindMesh =
                    eigenMatrixFromFbx(cluster->getTransformMatrix(), scale);
                const Eigen::Matrix4f bindBone =
                    eigenMatrixFromFbx(cluster->getTransformLinkMatrix(),
                                       scale);
                info.bindMeshMatrices.push_back(
                    rowMajorArrayFromEigen(bindMesh));
                info.bindBoneMatrices.push_back(
                    rowMajorArrayFromEigen(bindBone));
                info.skinnedMeshData.inverseBindMatrices.push_back(
                    glmMat4FromRowMajorArray(
                        rowMajorArrayFromEigen(bindBone.inverse())));

                const int indexCount = cluster->getIndicesCount();
                const int weightCount = cluster->getWeightsCount();
                if (indexCount != weightCount)
                    ++info.mismatchedClusterCount;

                const int count = std::min(indexCount, weightCount);
                const int* indices = cluster->getIndices();
                const double* weights = cluster->getWeights();
                if (!indices || !weights)
                    continue;
                for (int i = 0; i < count; ++i) {
                    controlPointInfluences[indices[i]].push_back(
                        {boneSlot, static_cast<float>(weights[i])});
                }
            }

            if (info.boneNames.size() > Scene::MaxSkinningBones) {
                throw std::runtime_error(fmt::format(
                    "FBX mesh '{}' has {} skin bones, but GPU skinning "
                    "supports at most {}",
                    info.mesh.name, info.boneNames.size(),
                    Scene::MaxSkinningBones));
            }

            for (size_t vertexIdx = 0; vertexIdx < renderToControlPoint.size();
                 ++vertexIdx) {
                const int controlPoint = renderToControlPoint[vertexIdx];
                auto influenceIt = controlPointInfluences.find(controlPoint);
                if (influenceIt == controlPointInfluences.end()) {
                    ++info.unweightedVertexCount;
                    continue;
                }

                bool droppedInfluence = false;
                for (const auto& [boneSlot, weight] : influenceIt->second) {
                    addVertexInfluence(
                        info.skinnedMeshData.boneIndices[vertexIdx],
                        info.skinnedMeshData.boneWeights[vertexIdx], boneSlot,
                        weight, droppedInfluence);
                }
                if (droppedInfluence)
                    ++info.overweightVertexCount;
                normalizeWeights(info.skinnedMeshData.boneWeights[vertexIdx]);
            }
        } else {
            info.unweightedVertexCount = info.mesh.vertexCount;
        }

        result.push_back(std::move(info));
    }

    return result;
}

} // namespace

SkeletonTree FBXLoader::loadSkeleton(const std::string& fbxPath, float scale) {
    const ofbx::u16 flags = static_cast<ofbx::u16>(
        ofbx::LoadFlags::IGNORE_GEOMETRY | ofbx::LoadFlags::IGNORE_MATERIALS |
        ofbx::LoadFlags::IGNORE_TEXTURES | ofbx::LoadFlags::IGNORE_ANIMATIONS |
        ofbx::LoadFlags::IGNORE_CAMERAS | ofbx::LoadFlags::IGNORE_LIGHTS);

    std::unique_ptr<ofbx::IScene, void (*)(ofbx::IScene*)> scene(
        loadSceneOrThrow(fbxPath, flags),
        [](ofbx::IScene* s) { s->destroy(); });

    const std::vector<const ofbx::Object*> ordered = orderedImportNodes(*scene);
    if (ordered.empty()) {
        throw std::runtime_error(
            fmt::format("FBX file '{}' has no importable transform nodes",
                        fbxPath));
    }

    std::vector<std::string> nodeNames;
    std::vector<int> parentIndices = parentIndicesForOrderedNodes(ordered);
    std::vector<Eigen::Vector3f> localTranslations;
    std::vector<Eigen::Quaternionf> localRotations;
    std::vector<int> numJointsInBody;

    nodeNames.reserve(ordered.size());
    localTranslations.reserve(ordered.size());
    localRotations.reserve(ordered.size());
    numJointsInBody.reserve(ordered.size());

    for (int i = 0; i < static_cast<int>(ordered.size()); ++i) {
        const ofbx::Object* object = ordered[i];

        const ofbx::DMatrix localTransform = object->getLocalTransform();
        nodeNames.push_back(makeNodeName(*object, i));
        localTranslations.push_back(translationFromMatrix(localTransform, scale));
        localRotations.push_back(rotationFromMatrix(localTransform));
        numJointsInBody.push_back(0);
    }

    return SkeletonTree(std::move(nodeNames), std::move(parentIndices),
                        std::move(localTranslations), std::move(localRotations),
                        std::move(numJointsInBody));
}

std::vector<FBXAnimationClipInfo>
FBXLoader::loadAnimationClipInfos(const std::string& fbxPath) {
    const ofbx::u16 flags = static_cast<ofbx::u16>(
        ofbx::LoadFlags::IGNORE_GEOMETRY | ofbx::LoadFlags::IGNORE_MATERIALS |
        ofbx::LoadFlags::IGNORE_TEXTURES | ofbx::LoadFlags::IGNORE_CAMERAS |
        ofbx::LoadFlags::IGNORE_LIGHTS);

    std::unique_ptr<ofbx::IScene, void (*)(ofbx::IScene*)> scene(
        loadSceneOrThrow(fbxPath, flags),
        [](ofbx::IScene* s) { s->destroy(); });

    return clipInfosFromScene(*scene);
}

SkeletonMotion FBXLoader::loadMotion(const std::string& fbxPath, int clipIndex,
                                     float fps, float scale) {
    const ofbx::u16 flags = static_cast<ofbx::u16>(
        ofbx::LoadFlags::IGNORE_GEOMETRY | ofbx::LoadFlags::IGNORE_MATERIALS |
        ofbx::LoadFlags::IGNORE_TEXTURES | ofbx::LoadFlags::IGNORE_CAMERAS |
        ofbx::LoadFlags::IGNORE_LIGHTS);

    std::unique_ptr<ofbx::IScene, void (*)(ofbx::IScene*)> scene(
        loadSceneOrThrow(fbxPath, flags),
        [](ofbx::IScene* s) { s->destroy(); });

    return loadMotionFromScene(fbxPath, *scene, clipIndex, fps, scale);
}

std::vector<FBXMeshInfo> FBXLoader::loadMeshes(const std::string& fbxPath,
                                               float scale) {
    const ofbx::u16 flags = static_cast<ofbx::u16>(
        ofbx::LoadFlags::IGNORE_ANIMATIONS | ofbx::LoadFlags::IGNORE_CAMERAS |
        ofbx::LoadFlags::IGNORE_LIGHTS);

    std::unique_ptr<ofbx::IScene, void (*)(ofbx::IScene*)> scene(
        loadSceneOrThrow(fbxPath, flags),
        [](ofbx::IScene* s) { s->destroy(); });

    std::vector<FBXMeshInfo> result;
    result.reserve(static_cast<size_t>(scene->getMeshCount()));

    for (int meshIdx = 0; meshIdx < scene->getMeshCount(); ++meshIdx) {
        const ofbx::Mesh* mesh = scene->getMesh(meshIdx);
        if (!mesh)
            continue;

        const ofbx::GeometryData& geom = mesh->getGeometryData();
        const ofbx::Vec3Attributes normals = geom.getNormals();
        const ofbx::Vec2Attributes uvs = geom.getUVs();

        FBXMeshInfo info;
        info.name = makeNodeName(*mesh, meshIdx);
        info.materialCount = mesh->getMaterialCount();
        info.materials = materialInfosForMesh(fbxPath, *mesh);
        if (!info.materials.empty())
            info.primaryMaterialIndex = 0;
        info.hasNormals = normals.values != nullptr;
        info.hasUVs = uvs.values != nullptr;

        const ofbx::Skin* skin = skinForMesh(*mesh);
        info.hasSkin = skin != nullptr;
        if (skin) {
            info.skinClusterNames.reserve(
                static_cast<size_t>(skin->getClusterCount()));
            for (int clusterIdx = 0; clusterIdx < skin->getClusterCount();
                 ++clusterIdx) {
                const ofbx::Cluster* cluster = skin->getCluster(clusterIdx);
                const ofbx::Object* link = cluster ? cluster->getLink() : nullptr;
                info.skinClusterNames.push_back(
                    link ? std::string(link->name)
                         : fmt::format("cluster_{}", clusterIdx));
            }
        }

        triangulateMeshGeometry(info.name, geom, scale, info.meshData);
        info.vertexCount = static_cast<int>(info.meshData.vertices.size());
        info.indexCount = static_cast<int>(info.meshData.indices.size());
        result.push_back(std::move(info));
    }

    return result;
}

std::vector<FBXSkinnedMeshInfo>
FBXLoader::loadSkinnedMeshes(const std::string& fbxPath, float scale) {
    const ofbx::u16 flags = static_cast<ofbx::u16>(
        ofbx::LoadFlags::IGNORE_ANIMATIONS | ofbx::LoadFlags::IGNORE_CAMERAS |
        ofbx::LoadFlags::IGNORE_LIGHTS);

    std::unique_ptr<ofbx::IScene, void (*)(ofbx::IScene*)> scene(
        loadSceneOrThrow(fbxPath, flags),
        [](ofbx::IScene* s) { s->destroy(); });

    return loadSkinnedMeshesFromScene(fbxPath, *scene, scale);
}

FBXCharacterAsset FBXLoader::loadCharacter(const std::string& fbxPath,
                                           int clipIndex, float fps,
                                           float scale) {
    const ofbx::u16 flags = static_cast<ofbx::u16>(
        ofbx::LoadFlags::IGNORE_CAMERAS | ofbx::LoadFlags::IGNORE_LIGHTS);

    std::unique_ptr<ofbx::IScene, void (*)(ofbx::IScene*)> scene(
        loadSceneOrThrow(fbxPath, flags),
        [](ofbx::IScene* s) { s->destroy(); });

    FBXCharacterAsset asset;
    asset.motion = loadMotionFromScene(fbxPath, *scene, clipIndex, fps, scale);
    asset.skinnedMeshes = loadSkinnedMeshesFromScene(fbxPath, *scene, scale);
    return asset;
}

FBXCharacterAsset FBXLoader::loadCharacterWithBind(
    const std::string& motionFbxPath, const std::string& bindFbxPath,
    int clipIndex, float fps, float scale) {
    FBXCharacterAsset asset;
    asset.motion = FBXLoader::loadMotion(motionFbxPath, clipIndex, fps, scale);
    asset.skinnedMeshes = FBXLoader::loadSkinnedMeshes(bindFbxPath, scale);
    bakeBindMeshTransforms(asset.skinnedMeshes);
    remapSkinnedMeshesToMotionByName(asset.skinnedMeshes, asset.motion);
    return asset;
}

std::vector<FBXSkinClusterInfo>
FBXLoader::loadSkinClusterInfos(const std::string& fbxPath, float scale) {
    const ofbx::u16 flags = static_cast<ofbx::u16>(
        ofbx::LoadFlags::IGNORE_ANIMATIONS | ofbx::LoadFlags::IGNORE_CAMERAS |
        ofbx::LoadFlags::IGNORE_LIGHTS);

    std::unique_ptr<ofbx::IScene, void (*)(ofbx::IScene*)> scene(
        loadSceneOrThrow(fbxPath, flags),
        [](ofbx::IScene* s) { s->destroy(); });

    std::vector<FBXSkinClusterInfo> result;
    for (int meshIdx = 0; meshIdx < scene->getMeshCount(); ++meshIdx) {
        const ofbx::Mesh* mesh = scene->getMesh(meshIdx);
        if (!mesh)
            continue;

        const ofbx::Skin* skin = skinForMesh(*mesh);
        if (!skin)
            continue;

        const std::string meshName = makeNodeName(*mesh, meshIdx);
        for (int clusterIdx = 0; clusterIdx < skin->getClusterCount();
             ++clusterIdx) {
            const ofbx::Cluster* cluster = skin->getCluster(clusterIdx);
            if (!cluster)
                continue;

            FBXSkinClusterInfo info;
            info.meshName = meshName;
            info.clusterName = makeNodeName(*cluster, clusterIdx);
            const ofbx::Object* link = cluster->getLink();
            info.linkName =
                link ? std::string(link->name)
                     : fmt::format("cluster_link_{}", clusterIdx);
            info.indexCount = cluster->getIndicesCount();
            info.weightCount = cluster->getWeightsCount();

            const int* indices = cluster->getIndices();
            const double* weights = cluster->getWeights();
            if (indices && info.indexCount > 0) {
                info.minIndex = indices[0];
                info.maxIndex = indices[0];
                for (int i = 1; i < info.indexCount; ++i) {
                    info.minIndex = std::min(info.minIndex, indices[i]);
                    info.maxIndex = std::max(info.maxIndex, indices[i]);
                }
            }
            if (weights && info.weightCount > 0) {
                info.minWeight = weights[0];
                info.maxWeight = weights[0];
                info.weightSum = 0.0;
                for (int i = 0; i < info.weightCount; ++i) {
                    info.minWeight = std::min(info.minWeight, weights[i]);
                    info.maxWeight = std::max(info.maxWeight, weights[i]);
                    info.weightSum += weights[i];
                }
            }

            const ofbx::DMatrix transform = cluster->getTransformMatrix();
            const ofbx::DMatrix linkTransform =
                cluster->getTransformLinkMatrix();
            info.transformTranslation[0] = transform.m[12] * scale;
            info.transformTranslation[1] = transform.m[13] * scale;
            info.transformTranslation[2] = transform.m[14] * scale;
            info.transformLinkTranslation[0] = linkTransform.m[12] * scale;
            info.transformLinkTranslation[1] = linkTransform.m[13] * scale;
            info.transformLinkTranslation[2] = linkTransform.m[14] * scale;

            result.push_back(std::move(info));
        }
    }

    return result;
}

} // namespace Animation
} // namespace KE
