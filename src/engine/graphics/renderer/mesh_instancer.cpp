#include "mesh_instancer.hpp"
#include "engine/graphics/material/material.hpp"
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vector>

namespace KE {

void MeshInstancer::init(Backend::GraphicsDevice* device,
                         Backend::Shader* shader, const Scene::MeshData& mesh,
                         PhongMaterial* material) {
    _device = device;
    _shader = shader;
    _material = material;

    _vao = device->createVertexArray();
    _vao->bind();

    _indexBuffer = device->createBuffer(
        Backend::BufferType::Index, sizeof(unsigned int) * mesh.indices.size(),
        mesh.indices.data());
    _vao->setIndexBuffer(_indexBuffer.get());
    _numIndices = (int)mesh.indices.size();

    auto addGeomAttr = [&](const auto& data, int location, int size) {
        if (data.empty())
            return;
        auto buf =
            device->createBuffer(Backend::BufferType::Vertex,
                                 sizeof(data[0]) * data.size(), data.data());
        _vao->setVertexBuffer(buf.get());
        Backend::VertexAttribute attr{
            location,        size, Backend::VertexAttributeType::Float, false,
            sizeof(data[0]), 0};
        _vao->setVertexAttribute(attr);
        _vbos.push_back(std::move(buf));
    };

    addGeomAttr(mesh.vertices, 0, 3);
    addGeomAttr(mesh.normals, 1, 3);
    addGeomAttr(mesh.uvs, 2, 2);

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
        Backend::VertexAttribute attr{3 + i,
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
    Backend::VertexAttribute colorAttr{
        7, 4, Backend::VertexAttributeType::Float, false, sizeof(glm::vec4),
        0, 1};
    _vao->setVertexAttribute(colorAttr);
}

void MeshInstancer::_reallocate(int newMax) {
    _allocatedInstances = newMax;
    _transformVBO = _device->createBuffer(Backend::BufferType::DynamicVertex,
                                          sizeof(glm::mat4) * newMax);
    _colorVBO = _device->createBuffer(Backend::BufferType::DynamicVertex,
                                      sizeof(glm::vec4) * newMax);
    _setupInstanceAttribs();
}

void MeshInstancer::setColors(const std::vector<glm::vec4>& colors) {
    int n = (int)colors.size();
    if (n > _allocatedInstances)
        _reallocate(n * 2);
    _colorVBO->setData(colors.data(), sizeof(glm::vec4) * n);
}

void MeshInstancer::updateFromTransforms(
    const std::vector<glm::mat4>& transforms,
    const std::vector<glm::vec4>* colors) {
    _visibleCount = (int)transforms.size();
    if (_visibleCount == 0)
        return;
    if (_visibleCount > _allocatedInstances)
        _reallocate(_visibleCount * 2);
    _transformVBO->setData(transforms.data(),
                           sizeof(glm::mat4) * _visibleCount);
    if (colors)
        _colorVBO->setData(colors->data(), sizeof(glm::vec4) * _visibleCount);
    _externalUpdate = true;
}

void MeshInstancer::update() {
    if (_externalUpdate) {
        _externalUpdate = false;
        return;
    }
    std::vector<glm::mat4> transforms;
    std::vector<glm::vec4> colors;
    _hasTransparent = false;

    for (auto* prim : _prims) {
        if (!prim || !prim->isVisible())
            continue;
        auto col = prim->getDisplayColorAlpha();
        glm::vec4 c = col ? *col : glm::vec4(1.f);
        if (c.a < 0.99f)
            _hasTransparent = true;
        transforms.push_back(prim->computeModelMatrix());
        colors.push_back(c);
    }

    _visibleCount = (int)transforms.size();
    if (_visibleCount == 0)
        return;

    if (_visibleCount > _allocatedInstances)
        _reallocate(_visibleCount * 2);

    _transformVBO->setData(transforms.data(),
                           sizeof(glm::mat4) * _visibleCount);
    _colorVBO->setData(colors.data(), sizeof(glm::vec4) * _visibleCount);
}

void MeshInstancer::render() {
    if (_visibleCount == 0)
        return;
    _vao->bind();
    _device->drawIndexedInstanced(_numIndices, _visibleCount);
    _vao->unbind();
}

} // namespace KE
