#include "rasterizer.hpp"
#include "backend/base/graphics_device.hpp"
#include "scene/native/prim.hpp"
#include "scene/scene_backend.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <memory>

namespace KE {

Rasterizer::Rasterizer(Backend::GraphicsDevice* graphicsDevice,
                       Scene::SceneBackend* scene) {
    _graphicsDevice = graphicsDevice;
    _scene = scene;
}

std::shared_ptr<Scene::ShapeRenderBuffer> Rasterizer::createRenderBuffer(
    Backend::Shader* shader, const std::shared_ptr<Scene::MeshData>& meshData) {
    auto bufferInfos = std::make_shared<Scene::ShapeRenderBuffer>();
    bufferInfos->backendShader = shader;

    // VAO
    bufferInfos->vertexArray = _graphicsDevice->createVertexArray();
    bufferInfos->vertexArray->bind();

    // Index Buffer
    bufferInfos->indexBuffer = _graphicsDevice->createBuffer(
        Backend::BufferType::Index,
        sizeof(unsigned int) * meshData->indices.size(),
        meshData->indices.data());

    bufferInfos->vertexArray->setIndexBuffer(
        bufferInfos->indexBuffer.get()); // bind

    // Helper to add attribute and keep buffer alive // TODO: use one buffer
    auto addAttribute = [&](const auto& data, int location, int size) {
        if (data.empty())
            return;
        auto buffer = _graphicsDevice->createBuffer(
            Backend::BufferType::Vertex, sizeof(data[0]) * data.size(),
            data.data());
        // buffer->bind();
        bufferInfos->vertexArray->setVertexBuffer(buffer.get());
        Backend::VertexAttribute attr = {
            location,        size, Backend::VertexAttributeType::Float, false,
            sizeof(data[0]), 0};
        bufferInfos->vertexArray->setVertexAttribute(attr);

        bufferInfos->vertexBuffers.push_back(std::move(buffer));
    };

    addAttribute(meshData->vertices, 0, 3);
    addAttribute(meshData->normals, 1, 3);
    addAttribute(meshData->uvs, 2, 2);

    bufferInfos->vertexArray->unbind();
    bufferInfos->numIndices = static_cast<int>(meshData->indices.size());

    return bufferInfos;
}

size_t Rasterizer::addShape(Backend::Shader* shader,
                            std::shared_ptr<Scene::MeshData> meshData) {
    if (!meshData || meshData->vertices.empty() || meshData->indices.empty()) {
        return static_cast<size_t>(-1); // Invalid shape ID
    }

    try {
        if (shader) {
            shader->use();
        }
        auto bufferInfos = createRenderBuffer(shader, meshData);

        _renderList.push_back(bufferInfos);
        return _renderList.size() - 1;

    } catch (const std::exception& e) {
        std::cerr << "Failed to add shape with Scene::MeshData: " << e.what()
                  << std::endl;
        return static_cast<size_t>(-1);
    }
}

size_t Rasterizer::addShape(PhongMaterial* material,
                            std::shared_ptr<Scene::MeshData> meshData) {
    if (!meshData || meshData->vertices.empty() || meshData->indices.empty()) {
        return static_cast<size_t>(-1); // Invalid shape ID
    }

    try {
        Backend::Shader* shader = material->getShader();

        if (shader) {
            material->bind();
        }

        auto bufferInfos = createRenderBuffer(shader, meshData);

        _renderList.push_back(bufferInfos);
        return _renderList.size() - 1;
    } catch (const std::exception& e) {
        std::cerr << "Failed to add shape with PhongMaterial: " << e.what()
                  << std::endl;
        return static_cast<size_t>(-1);
    }
}

// Shader + Prim (non-owning: scene graph owns the Prim)
size_t Rasterizer::addShape(Backend::Shader* shader, Scene::Prim* prim) {

    std::shared_ptr<Scene::MeshData> meshData = prim->getMeshData();
    if (!meshData || meshData->vertices.empty() || meshData->indices.empty()) {
        return static_cast<size_t>(-1); // Invalid shape ID
    }

    try {
        if (shader) {
            shader->use();
        }

        auto bufferInfos = createRenderBuffer(shader, meshData);
        bufferInfos->prim = prim;

        _renderList.push_back(bufferInfos);
        return _renderList.size() - 1;

    } catch (const std::exception& e) {
        std::cerr << "Failed to add shape with Scene::Prim: " << e.what()
                  << std::endl;
        return static_cast<size_t>(-1);
    }
}

void Rasterizer::render(const glm::mat4& view, const glm::mat4& proj) {
    Backend::Shader* lastShader = nullptr;

    auto drawBuffer = [&](const auto& buffer) {
        if (lastShader != buffer->backendShader) {
            buffer->backendShader->use();
            buffer->backendShader->setMat4("view", view);
            buffer->backendShader->setMat4("projection", proj);
            lastShader = buffer->backendShader;
        }

        if (buffer->prim) {
            // fmt::print("prim address: {}, hasTranslate: {}\n",
            //             (void*)buffer->prim.get(),
            //             buffer->prim->hasAttribute("xformOp:translate"));
            auto model = buffer->prim->computeModelMatrix();
            buffer->backendShader->setMat4("model", model);
            buffer->backendShader->setMat3(
                "normalMat",
                glm::mat3(glm::transpose(glm::inverse(view * model))));
            if (auto color = buffer->prim->getDisplayColorAlpha())
                buffer->backendShader->setVec4("objColor", *color);
        }
        buffer->vertexArray->bind();
        _graphicsDevice->drawIndexed(buffer->numIndices);
        buffer->vertexArray->unbind();
    };

    // TODO: implement Transparency in material.
    auto isTransparent = [](const auto& buffer) -> bool {
        if (!buffer->prim)
            return false;
        auto col = buffer->prim->getDisplayColorAlpha();
        return col && (col->a < 0.99f);
    };

    // Render Pass 1, opaque
    for (const auto& buffer : _renderList) {
        if (!buffer || !buffer->backendShader)
            continue;
        if (buffer->prim && !buffer->prim->isVisible())
            continue;
        if (isTransparent(buffer))
            continue;
        drawBuffer(buffer);
    }

    // Render Pass 2: transparent, back-to-front sorted by view-space z
    // View space: camera looks toward -z, more negative = further away
    using Entry = std::pair<float, Scene::ShapeRenderBuffer*>;
    std::vector<Entry> transparents;
    for (const auto& buffer : _renderList) {
        if (!buffer || !buffer->backendShader)
            continue;
        if (buffer->prim && !buffer->prim->isVisible())
            continue;
        if (!isTransparent(buffer))
            continue;
        glm::vec4 worldPos = buffer->prim->computeModelMatrix()[3];
        float viewZ = (view * worldPos).z;
        transparents.emplace_back(viewZ, buffer.get());
    }
    // ascending sort: most negative z (furthest) first → back-to-front
    std::sort(transparents.begin(), transparents.end(),
              [](const Entry& a, const Entry& b) { return a.first < b.first; });

    _graphicsDevice->setBlend(true);
    _graphicsDevice->setBlendFunc(Backend::BlendFactor::SrcAlpha,
                                  Backend::BlendFactor::OneMinusSrcAlpha);
    _graphicsDevice->setDepthWrite(false);
    for (auto& [z, buf] : transparents)
        drawBuffer(buf);
    _graphicsDevice->setDepthWrite(true);
    _graphicsDevice->setBlend(false);
}

/*
void Rasterizer::renderOutline(const glm::mat4& view, const glm::mat4& proj,
                               Backend::Shader* outlineShader, float thickness,
                               const glm::vec4& color) {
    if (!outlineShader)
        return;
    outlineShader->use();
    outlineShader->setMat4("view", view);
    outlineShader->setMat4("projection", proj);
    outlineShader->setFloat("outlineThickness", thickness);
    outlineShader->setVec4("outlineColor", color);
    for (const auto& buffer : _renderList) {
        if (!buffer || !buffer->prim || !buffer->outlined)
            continue;
        auto model = buffer->prim->computeModelMatrix();
        outlineShader->setMat4("model", model);
        buffer->vertexArray->bind();
        _graphicsDevice->drawIndexed(buffer->numIndices);
        buffer->vertexArray->unbind();
    }
}

void Rasterizer::setOutlined(size_t idx, bool outlined) {
    if (idx < _renderList.size())
        _renderList[idx]->outlined = outlined;
}*/

} // namespace KE