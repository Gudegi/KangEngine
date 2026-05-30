#include "mesh_instancer.hpp"
#include "engine/graphics/material/material.hpp"
#include "geometry/mesh_utils.hpp"
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace KE {

void MeshInstancer::_initMeshData(const Scene::MeshData& mesh) {
    _localBounds = mesh.computeLocalAABB();
    // Reserved for future sphere culling / LOD paths; current culling uses
    // AABB.
    _localSphere = Geometry::computeBoundingSphere(_localBounds);

    Scene::MeshData tangentMesh;
    const Scene::MeshData* uploadMesh = &mesh;
    if (mesh.tangents.empty() && !mesh.vertices.empty() &&
        mesh.normals.size() == mesh.vertices.size() &&
        mesh.uvs.size() == mesh.vertices.size()) {
        tangentMesh = mesh;
        Geometry::computeTangents(tangentMesh);
        uploadMesh = &tangentMesh;
    }

    _indexBuffer = _device->createBuffer(
        Backend::BufferType::Index,
        sizeof(unsigned int) * uploadMesh->indices.size(),
        uploadMesh->indices.data());
    _vao->setIndexBuffer(_indexBuffer.get());
    _numIndices = (int)uploadMesh->indices.size();
    _hasTangents = uploadMesh->tangents.size() == uploadMesh->vertices.size();

    auto addGeomAttr = [&](const auto& data, int location, int size) {
        if (data.empty())
            return;
        auto buf =
            _device->createBuffer(Backend::BufferType::Vertex,
                                  sizeof(data[0]) * data.size(), data.data());
        _vao->setVertexBuffer(buf.get());
        Backend::VertexAttribute attr{
            location,        size, Backend::VertexAttributeType::Float, false,
            sizeof(data[0]), 0};
        _vao->setVertexAttribute(attr);
        _vbos.push_back(std::move(buf));
    };

    addGeomAttr(uploadMesh->vertices, RendererAttribute::Position, 3);
    addGeomAttr(uploadMesh->normals, RendererAttribute::Normal, 3);
    addGeomAttr(uploadMesh->uvs, RendererAttribute::TexCoord, 2);
    if (_hasTangents)
        addGeomAttr(uploadMesh->tangents, RendererAttribute::Tangent, 4);
}

void MeshInstancer::_setupSkinningAttribs(
    const Scene::SkinnedMeshData& skinnedMesh) {
    if (!skinnedMesh.hasValidVertexSkinning())
        return;
    if (skinnedMesh.inverseBindMatrices.size() > Scene::MaxSkinningBones ||
        skinnedMesh.boneNodeIndices.size() > Scene::MaxSkinningBones) {
        throw std::runtime_error(
            "Skinned mesh has " +
            std::to_string(std::max(skinnedMesh.inverseBindMatrices.size(),
                                    skinnedMesh.boneNodeIndices.size())) +
            " bones, but GPU skinning supports at most " +
            std::to_string(Scene::MaxSkinningBones));
    }
    _hasSkinning = true;

    auto boneIndexBuffer = _device->createBuffer(
        Backend::BufferType::Vertex,
        sizeof(skinnedMesh.boneIndices[0]) * skinnedMesh.boneIndices.size(),
        skinnedMesh.boneIndices.data());
    _vao->setVertexBuffer(boneIndexBuffer.get());
    Backend::VertexAttribute boneIndexAttr{
        RendererAttribute::BoneIndices,     4,
        Backend::VertexAttributeType::Int,  false,
        sizeof(skinnedMesh.boneIndices[0]), 0};
    _vao->setVertexAttribute(boneIndexAttr);
    _vbos.push_back(std::move(boneIndexBuffer));

    auto boneWeightBuffer = _device->createBuffer(
        Backend::BufferType::Vertex,
        sizeof(skinnedMesh.boneWeights[0]) * skinnedMesh.boneWeights.size(),
        skinnedMesh.boneWeights.data());
    _vao->setVertexBuffer(boneWeightBuffer.get());
    Backend::VertexAttribute boneWeightAttr{
        RendererAttribute::BoneWeights,      4,
        Backend::VertexAttributeType::Float, false,
        sizeof(skinnedMesh.boneWeights[0]),  0};
    _vao->setVertexAttribute(boneWeightAttr);
    _vbos.push_back(std::move(boneWeightBuffer));
}

void MeshInstancer::init(Backend::GraphicsDevice* device,
                         Backend::Shader* shader, const Scene::MeshData& mesh,
                         PhongMaterial* material) {
    _device = device;
    _shader = shader;
    _material = material;

    _vao = device->createVertexArray();
    _vao->bind();

    _initMeshData(mesh);

    _vao->unbind();
}

void MeshInstancer::init(Backend::GraphicsDevice* device,
                         Backend::Shader* shader,
                         const Scene::SkinnedMeshData& skinnedMesh,
                         PhongMaterial* material) {
    _device = device;
    _shader = shader;
    _material = material;

    _vao = device->createVertexArray();
    _vao->bind();

    _initMeshData(skinnedMesh.mesh);
    _setupSkinningAttribs(skinnedMesh);

    _vao->unbind();
}

void MeshInstancer::addPrim(Scene::Prim* prim) { _prims.push_back(prim); }

void MeshInstancer::removePrim(Scene::Prim* prim) {
    _prims.erase(std::remove(_prims.begin(), _prims.end(), prim), _prims.end());
}

void MeshInstancer::_setupInstanceAttribs() {
    // Transform mat4: locations 3-6 (4 × vec4, stride = sizeof(mat4))
    _vao->setVertexBuffer(_transformVBO.get());
    for (int i = 0; i < 4; i++) {
        Backend::VertexAttribute attr{RendererAttribute::InstanceTransform0 + i,
                                      4,
                                      Backend::VertexAttributeType::Float,
                                      false,
                                      sizeof(glm::mat4),
                                      (size_t)(i * sizeof(glm::vec4)),
                                      1};
        _vao->setVertexAttribute(attr);
    }

    // Color vec4: location 7
    _vao->setVertexBuffer(_colorVBO.get());
    Backend::VertexAttribute colorAttr{RendererAttribute::InstanceColor,
                                       4,
                                       Backend::VertexAttributeType::Float,
                                       false,
                                       sizeof(glm::vec4),
                                       0,
                                       1};
    _vao->setVertexAttribute(colorAttr);
}

void MeshInstancer::uploadSkinningMatrices(Backend::Shader* shader) {
    Backend::Shader* targetShader = shader ? shader : _shader;
    if (!targetShader || _boneMatrices.empty())
        return;
    if (_boneMatrices.size() > Scene::MaxSkinningBones) {
        throw std::runtime_error(
            "Skinned mesh upload has " + std::to_string(_boneMatrices.size()) +
            " bone matrices, but GPU skinning supports at most " +
            std::to_string(Scene::MaxSkinningBones));
    }

    targetShader->setMat4Array("uBoneMatrices[0]", _boneMatrices.data(),
                               _boneMatrices.size());
}

void MeshInstancer::_reallocate(int newMax) {
    _allocatedInstances = newMax;
    _transformVBO = _device->createBuffer(Backend::BufferType::DynamicVertex,
                                          sizeof(glm::mat4) * newMax);
    _colorVBO = _device->createBuffer(Backend::BufferType::DynamicVertex,
                                      sizeof(glm::vec4) * newMax);
    _setupInstanceAttribs();
}

void MeshInstancer::_updateWorldBounds(
    const std::vector<glm::mat4>& transforms) {
    _worldBounds.clear();
    _combinedWorldBounds = Geometry::AABB::empty();

    if (!_localBounds.isValid())
        return;

    if (_worldBounds.capacity() < transforms.size())
        _worldBounds.reserve(transforms.size());
    for (const glm::mat4& transform : transforms) {
        Geometry::AABB world = Geometry::transformAABB(_localBounds, transform);
        _worldBounds.push_back(world);
        _combinedWorldBounds.expand(world);
    }
}

void MeshInstancer::_uploadInstanceData(
    const std::vector<glm::mat4>& transforms,
    const std::vector<glm::vec4>& colors) {
    _visibleCount = static_cast<int>(transforms.size());
    if (_visibleCount == 0)
        return;

    if (_visibleCount > _allocatedInstances)
        _reallocate(_visibleCount * 2);

    _transformVBO->setData(transforms.data(),
                           sizeof(glm::mat4) * _visibleCount);
    if (!colors.empty())
        _colorVBO->setData(colors.data(), sizeof(glm::vec4) * _visibleCount);
}

void MeshInstancer::_updateTransparency() {
    _hasTransparent = false;
    for (const auto& color : _colors) {
        if (color.a < 1.0f) {
            _hasTransparent = true;
            break;
        }
    }
}

void MeshInstancer::setColors(const std::vector<glm::vec4>& colors) {
    _colors = colors;
    _updateTransparency();
    _uploadInstanceData(_transforms, _colors);
}

void MeshInstancer::updateFromTransforms(
    const std::vector<glm::mat4>& transforms,
    const std::vector<glm::vec4>* colors) {
    _transforms = transforms;
    if (colors)
        _colors = *colors;
    if (_colors.size() != _transforms.size())
        _colors.assign(_transforms.size(), glm::vec4(1.0f));

    _updateTransparency();
    _updateWorldBounds(_transforms);
    _uploadInstanceData(_transforms, _colors);
    _useExternalTransforms = true;
}

void MeshInstancer::update() {
    if (_useExternalTransforms) {
        _uploadInstanceData(_transforms, _colors);
        return;
    }

    _transforms.clear();
    _colors.clear();
    if (_transforms.capacity() < _prims.size())
        _transforms.reserve(_prims.size());
    if (_colors.capacity() < _prims.size())
        _colors.reserve(_prims.size());

    for (auto* prim : _prims) {
        if (!prim || !prim->isVisible())
            continue;
        auto col = prim->getDisplayColorAlpha();
        glm::vec4 c = col ? *col : glm::vec4(1.f);
        _transforms.push_back(prim->computeModelMatrix());
        _colors.push_back(c);
    }

    _updateTransparency();
    _updateWorldBounds(_transforms);
    _uploadInstanceData(_transforms, _colors);
}

void MeshInstancer::updateGeometry(const std::vector<glm::vec3>& positions,
                                   const std::vector<glm::vec3>& normals) {
    // TODO: Soft-body/deformable meshes need world bounds refresh after this.
    // _vbos[0] = positions (location 0), _vbos[1] = normals (location 1)
    if (!_vbos.empty())
        _vbos[0]->setData(positions.data(),
                          sizeof(glm::vec3) * positions.size());
    if (_vbos.size() > 1 && !normals.empty())
        _vbos[1]->setData(normals.data(), sizeof(glm::vec3) * normals.size());
    _localBounds = Geometry::computeAABB(positions);
    _localSphere = Geometry::computeBoundingSphere(_localBounds);
}

void MeshInstancer::updateSkinningMatrices(
    const std::vector<glm::mat4>& boneMatrices) {
    if (!_hasSkinning)
        return;
    _boneMatrices = boneMatrices;
}

void MeshInstancer::applyFrustumCulling(const Geometry::Frustum* frustum) {
    if (!frustum || _hasSkinning || _worldBounds.size() != _transforms.size()) {
        _uploadInstanceData(_transforms, _colors);
        return;
    }

    _culledTransforms.clear();
    _culledColors.clear();
    if (_culledTransforms.capacity() < _transforms.size())
        _culledTransforms.reserve(_transforms.size());
    if (_culledColors.capacity() < _colors.size())
        _culledColors.reserve(_colors.size());

    for (size_t i = 0; i < _transforms.size(); ++i) {
        if (!Geometry::intersects(*frustum, _worldBounds[i]))
            continue;
        _culledTransforms.push_back(_transforms[i]);
        _culledColors.push_back(i < _colors.size() ? _colors[i]
                                                   : glm::vec4(1.0f));
    }

    _uploadInstanceData(_culledTransforms, _culledColors);
}

void MeshInstancer::render() {
    if (_visibleCount == 0)
        return;
    _vao->bind();
    _device->drawIndexedInstanced(_numIndices, _visibleCount);
    _vao->unbind();
}

} // namespace KE
