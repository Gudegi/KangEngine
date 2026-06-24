#include "renderer.hpp"

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/camera/camera.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"

#include <utility>

namespace KE {

void Renderer::bind(Backend::GraphicsDevice* device, Rasterizer* rasterizer,
                    PostProcessor* postProcessor,
                    SelectionOutlineProcessor* selectionOutlineProcessor) {
    _device = device;
    _rasterizer = rasterizer;
    _postProcessor = postProcessor;
    _selectionOutlineProcessor = selectionOutlineProcessor;
}

void Renderer::setViewportSize(int width, int height) {
    _viewportWidth = width;
    _viewportHeight = height;
}

void Renderer::setLight(const DirectionalLight& light) {
    if (_rasterizer)
        _rasterizer->setLight(light);
}

const DirectionalLight& Renderer::light() const {
    return _rasterizer->getLight();
}

void Renderer::setPointLights(std::vector<PointLight> lights) {
    if (_rasterizer)
        _rasterizer->setPointLights(std::move(lights));
}

const std::vector<PointLight>& Renderer::pointLights() const {
    return _rasterizer->getPointLights();
}

void Renderer::setSpotLights(std::vector<SpotLight> lights) {
    if (_rasterizer)
        _rasterizer->setSpotLights(std::move(lights));
}

const std::vector<SpotLight>& Renderer::spotLights() const {
    return _rasterizer->getSpotLights();
}

Backend::Framebuffer* Renderer::shadowFbo() {
    return _rasterizer ? _rasterizer->getShadowFbo() : nullptr;
}

void Renderer::renderSceneToFramebuffer(Camera& camera,
                                        Backend::Framebuffer* target, int width,
                                        int height, bool clear) {
    if (!_rasterizer || !_device || !target || width <= 0 || height <= 0)
        return;

    const glm::mat4 view = camera.getViewMatrix();
    const glm::mat4 proj = camera.getProjMatrix();

    _rasterizer->updateFrameData(view, proj);
    target->bind();
    _device->setViewport(0, 0, width, height);
    if (clear)
        _device->clear(0.2f, 0.3f, 0.3f, 1.0f);
    _rasterizer->render(view, proj);
    _device->setPolygonMode(Backend::PolygonMode::Fill);
    target->resolve();
    target->unbind();
    if (_viewportWidth > 0 && _viewportHeight > 0)
        _device->setViewport(0, 0, _viewportWidth, _viewportHeight);
}

void Renderer::updateRenderableTransforms(
    RenderableHandle handle, const std::vector<glm::mat4>& transforms,
    const std::vector<glm::vec4>* colors) {
    if (_rasterizer)
        _rasterizer->updateRenderableTransforms(handle, transforms, colors);
}

bool Renderer::getRenderableInstanceTransform(RenderableHandle handle,
                                              int instanceIndex,
                                              glm::mat4& outTransform) const {
    return _rasterizer && _rasterizer->getRenderableInstanceTransform(
                              handle, instanceIndex, outTransform);
}

bool Renderer::setRenderableInstanceTransform(RenderableHandle handle,
                                              int instanceIndex,
                                              const glm::mat4& transform) {
    return _rasterizer && _rasterizer->setRenderableInstanceTransform(
                              handle, instanceIndex, transform);
}

void Renderer::setRenderableColors(RenderableHandle handle,
                                   const std::vector<glm::vec4>& colors) {
    if (_rasterizer)
        _rasterizer->setRenderableColors(handle, colors);
}

void Renderer::setRenderableDoubleSided(RenderableHandle handle,
                                        bool doubleSided) {
    if (_rasterizer)
        _rasterizer->setRenderableDoubleSided(handle, doubleSided);
}

void Renderer::setRenderableCastsShadow(RenderableHandle handle,
                                        bool castsShadow) {
    if (_rasterizer)
        _rasterizer->setRenderableCastsShadow(handle, castsShadow);
}

void Renderer::setRenderableTexture(RenderableHandle handle,
                                    Backend::Texture* tex, TextureRole role) {
    setRenderableTexture(handle, tex, textureRoleSlot(role));
}

void Renderer::setRenderableTexture(RenderableHandle handle,
                                    Backend::Texture* tex, int slot) {
    if (_rasterizer)
        _rasterizer->setRenderableTexture(handle, tex, slot);
}

void Renderer::updateRenderableGeometry(RenderableHandle handle,
                                        const std::vector<glm::vec3>& positions,
                                        const std::vector<glm::vec3>& normals) {
    if (_rasterizer)
        _rasterizer->updateRenderableGeometry(handle, positions, normals);
}

void Renderer::updateRenderableSkinningMatrices(
    RenderableHandle handle, const std::vector<glm::mat4>& boneMatrices) {
    if (_rasterizer)
        _rasterizer->updateRenderableSkinningMatrices(handle, boneMatrices);
}

} // namespace KE
