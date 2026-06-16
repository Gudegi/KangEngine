#include "selection_outline_processor.hpp"

namespace KE {

namespace {

static const char* SelectionOutlineVs = R"(
#version 410 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aUV;
}
)";

static const char* SelectionOutlineFs = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D uScene;
uniform sampler2D uMask;
uniform vec2 uTexelSize;
uniform vec4 uOutlineColor;
uniform float uOutlineRadius;

void main() {
    vec4 sceneColor = texture(uScene, TexCoord);
    float center = texture(uMask, TexCoord).r;
    float neighbor = 0.0;

    int radius = int(clamp(ceil(uOutlineRadius), 1.0, 8.0));
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            vec2 offset = vec2(float(x), float(y)) * uTexelSize;
            vec2 sampleUv = clamp(TexCoord + offset, vec2(0.0), vec2(1.0));
            neighbor = max(neighbor, texture(uMask, sampleUv).r);
        }
    }

    float outline = step(0.5, neighbor) * (1.0 - step(0.5, center));
    vec3 color = mix(sceneColor.rgb, uOutlineColor.rgb, outline * uOutlineColor.a);
    FragColor = vec4(color, sceneColor.a);
}
)";

static const float QuadPos[] = {
    -1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 1.f, 1.f,
};
static const float QuadUV[] = {
    0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 1.f,
};
static const unsigned int QuadIdx[] = {0, 1, 2, 0, 2, 3};

} // namespace

void SelectionOutlineProcessor::init(Backend::GraphicsDevice* device, int width,
                                     int height) {
    _device = device;
    _width = width;
    _height = height;

    _shader = device->createShader(SelectionOutlineVs, SelectionOutlineFs);
    _posVBO = device->createBuffer(Backend::BufferType::Vertex, sizeof(QuadPos),
                                   QuadPos);
    _uvVBO = device->createBuffer(Backend::BufferType::Vertex, sizeof(QuadUV),
                                  QuadUV);
    _ibo = device->createBuffer(Backend::BufferType::Index, sizeof(QuadIdx),
                                QuadIdx);

    _quadVAO = device->createVertexArray();
    _quadVAO->bind();
    _quadVAO->setIndexBuffer(_ibo.get());
    _quadVAO->setVertexBuffer(_posVBO.get());
    _quadVAO->setVertexAttribute({0, 2, Backend::VertexAttributeType::Float,
                                  false, 2 * sizeof(float), 0});
    _quadVAO->setVertexBuffer(_uvVBO.get());
    _quadVAO->setVertexAttribute({1, 2, Backend::VertexAttributeType::Float,
                                  false, 2 * sizeof(float), 0});
    _quadVAO->unbind();
    // Output format must be 16f for HDR.
    _outputFBO =
        device->createFramebuffer({width, height, false, false, 0,
                                   Backend::FramebufferColorFormat::RGBA16F});
}

void SelectionOutlineProcessor::renderOutlineCompositePass(
    Backend::Texture* sceneColor, Backend::Texture* selectionMask) {
    if (!sceneColor || !selectionMask)
        return;

    _outputFBO->bind();
    _device->setViewport(0, 0, _width, _height);
    _device->clear(0.f, 0.f, 0.f, 1.f);
    _device->setDepthTest(false);

    _shader->use();
    _shader->setInt("uScene", 0);
    _shader->setInt("uMask", 1);
    _shader->setVec2("uTexelSize", 1.0f / static_cast<float>(_width),
                     1.0f / static_cast<float>(_height));
    _shader->setVec4("uOutlineColor", _config.color);
    _shader->setFloat("uOutlineRadius", _config.radius);
    sceneColor->bind(0);
    selectionMask->bind(1);

    _quadVAO->bind();
    _device->drawIndexed(6);
    _quadVAO->unbind();

    _device->setDepthTest(true);
    _outputFBO->unbind();
}

Backend::Texture* SelectionOutlineProcessor::getResult() {
    return _outputFBO->getColorTexture();
}

Backend::Framebuffer* SelectionOutlineProcessor::getOutputFramebuffer() {
    return _outputFBO.get();
}

void SelectionOutlineProcessor::blitToScreen(int width, int height) {
    _outputFBO->blitToScreen(width, height);
}

void SelectionOutlineProcessor::resize(int width, int height) {
    _width = width;
    _height = height;
    _outputFBO->resize(width, height);
}

} // namespace KE
