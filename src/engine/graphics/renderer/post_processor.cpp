#include "post_processor.hpp"

namespace KE {

static const char* GLpostVs = R"(
#version 410 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aUV;
}
)";

static const char* GLpostFs = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uScreen;
uniform float uGamma;
void main() {
    vec4 color = texture(uScreen, TexCoord);
    FragColor = vec4(pow(color.rgb, vec3(1.0 / uGamma)), color.a);
}
)";

static const float quadPos[] = {
    -1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 1.f, 1.f,
};
static const float quadUV[] = {
    0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 1.f,
};
static const unsigned int quadIdx[] = {0, 1, 2, 0, 2, 3};

void PostProcessor::init(Backend::GraphicsDevice* device, int width,
                         int height) {
    _device = device;
    _width = width;
    _height = height;

    _shader = device->createShader(GLpostVs, GLpostFs);

    _posVBO = device->createBuffer(Backend::BufferType::Vertex, sizeof(quadPos),
                                   quadPos);
    _uvVBO = device->createBuffer(Backend::BufferType::Vertex, sizeof(quadUV),
                                  quadUV);
    _ibo = device->createBuffer(Backend::BufferType::Index, sizeof(quadIdx),
                                quadIdx);

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

    _outputFBO = device->createFramebuffer({width, height, false, false, 0});
}

void PostProcessor::process(Backend::Texture* src, float gamma) {
    _outputFBO->bind();
    _device->clear(0.f, 0.f, 0.f, 1.f);
    _device->setDepthTest(false);

    _shader->use();
    _shader->setInt("uScreen", 0);
    _shader->setFloat("uGamma", gamma < 0.01f ? 1.f : gamma);
    src->bind(0);

    _quadVAO->bind();
    _device->drawIndexed(6);
    _quadVAO->unbind();

    _device->setDepthTest(true);
    _outputFBO->unbind();
}

Backend::Texture* PostProcessor::getResult() {
    return _outputFBO->getColorTexture();
}

void PostProcessor::blitToScreen(int width, int height) {
    _outputFBO->blitToScreen(width, height);
}

void PostProcessor::resize(int width, int height) {
    _width = width;
    _height = height;
    _outputFBO->resize(width, height);
}

} // namespace KE
