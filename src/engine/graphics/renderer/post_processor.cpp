#include "post_processor.hpp"
#include <algorithm>

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
uniform int uToneMapMode;
uniform float uToneMapExposure;

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 acesNarkowicz(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Stephen Hill / MJP BakingLab ACES fit.
// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
vec3 rrtAndOdtFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 acesFitted(vec3 color) {
    const mat3 acesInputMat = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    const mat3 acesOutputMat = mat3(
         1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );

    color = acesInputMat * color;
    color = rrtAndOdtFit(color);
    color = acesOutputMat * color;
    return clamp(color, 0.0, 1.0);
}

void main() {
    vec4 color = texture(uScreen, TexCoord);

    vec3 mapped = color.rgb;
    if (uToneMapMode == 1) {
        mapped = mapped * uToneMapExposure;
        mapped = mapped / (mapped + vec3(1.0));
    } else if (uToneMapMode == 2) {
        mapped = vec3(1.0) - exp(-mapped * uToneMapExposure);
    } else if (uToneMapMode == 3) {
        mapped = acesNarkowicz(mapped * uToneMapExposure);
    } else if (uToneMapMode == 4) {
        mapped = acesFitted(mapped * uToneMapExposure);
    }

    vec3 corrected = pow(max(mapped, vec3(0.0)), vec3(1.0 / uGamma));
    FragColor = vec4(corrected, color.a);
}
)";

static const char* GLbrightExtractFs = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D uScene;
uniform float uThreshold;

void main() {
    vec4 color = texture(uScene, TexCoord);
    float brightness = max(max(color.r, color.g), color.b);
    vec3 bright = brightness > uThreshold ? color.rgb : vec3(0.0);
    FragColor = vec4(bright, color.a);
}
)";

static const char* GLblurFs = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D uImage;
uniform bool uHorizontal;

void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(uImage, 0));
    vec3 result = texture(uImage, TexCoord).rgb * 0.227027;

    float weights[4] = float[4](0.1945946, 0.1216216, 0.054054, 0.016216);
    for (int i = 0; i < 4; ++i) {
        float offset = float(i + 1);
        vec2 delta = uHorizontal
            ? vec2(texelSize.x * offset, 0.0)
            : vec2(0.0, texelSize.y * offset);
        result += texture(uImage, TexCoord + delta).rgb * weights[i];
        result += texture(uImage, TexCoord - delta).rgb * weights[i];
    }

    FragColor = vec4(result, 1.0);
}
)";

static const char* GLbloomCompositeFs = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomIntensity;

void main() {
    vec4 scene = texture(uScene, TexCoord);
    vec3 bloom = texture(uBloom, TexCoord).rgb * uBloomIntensity;
    FragColor = vec4(scene.rgb + bloom, scene.a);
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

    _toneMapShader = device->createShader(GLpostVs, GLpostFs);
    _brightExtractShader = device->createShader(GLpostVs, GLbrightExtractFs);
    _blurShader = device->createShader(GLpostVs, GLblurFs);
    _bloomCompositeShader = device->createShader(GLpostVs, GLbloomCompositeFs);

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

void PostProcessor::process(Backend::Texture* src, float gamma,
                            ToneMapMode toneMapMode, float tonemapExposure,
                            const BloomConfig& bloom) {
    if (!src)
        return;

    Backend::Texture* HDRSource = src;
    if (bloom.enabled) {
        ensureBloomBuffers(bloom);
        renderBrightExtractPass(src, _bloomPingPongFBO[0].get(),
                                bloom.threshold);
        Backend::Texture* blurredBloom =
            renderBloomBlurPass(_bloomPingPongFBO[0]->getColorTexture(), bloom);
        renderBloomCompositePass(src, blurredBloom, _bloomCompositeFBO.get(),
                                 bloom.intensity);
        HDRSource = _bloomCompositeFBO->getColorTexture();
    }

    renderToneMapPass(HDRSource, gamma, toneMapMode, tonemapExposure);
}

void PostProcessor::ensureBloomBuffers(const BloomConfig& bloom) {
    const int downsample = std::max(1, bloom.downsample);
    const int width = std::max(1, _width / downsample);
    const int height = std::max(1, _height / downsample);

    if (!_bloomPingPongFBO[0] || !_bloomPingPongFBO[1] || !_bloomCompositeFBO) {
        Backend::FramebufferDesc bloomDesc;
        bloomDesc.width = width;
        bloomDesc.height = height;
        bloomDesc.colorFormat = Backend::FramebufferColorFormat::RGBA16F;
        _bloomPingPongFBO[0] = _device->createFramebuffer(bloomDesc);
        _bloomPingPongFBO[1] = _device->createFramebuffer(bloomDesc);

        Backend::FramebufferDesc compositeDesc;
        compositeDesc.width = _width;
        compositeDesc.height = _height;
        compositeDesc.colorFormat = Backend::FramebufferColorFormat::RGBA16F;
        _bloomCompositeFBO = _device->createFramebuffer(compositeDesc);
        _bloomWidth = width;
        _bloomHeight = height;
        _bloomDownsample = downsample;
        return;
    }

    if (_bloomWidth != width || _bloomHeight != height ||
        _bloomDownsample != downsample) {
        _bloomPingPongFBO[0]->resize(width, height);
        _bloomPingPongFBO[1]->resize(width, height);
        _bloomWidth = width;
        _bloomHeight = height;
        _bloomDownsample = downsample;
    }

    _bloomCompositeFBO->resize(_width, _height);
}

void PostProcessor::drawFullscreen() {
    _quadVAO->bind();
    _device->drawIndexed(6);
    _quadVAO->unbind();
}

void PostProcessor::renderBrightExtractPass(Backend::Texture* src,
                                            Backend::Framebuffer* target,
                                            float threshold) {
    target->bind();
    _device->setViewport(0, 0, _bloomWidth, _bloomHeight);
    _device->clear(0.f, 0.f, 0.f, 1.f);
    _device->setDepthTest(false);

    _brightExtractShader->use();
    _brightExtractShader->setInt("uScene", 0);
    _brightExtractShader->setFloat("uThreshold", threshold);
    src->bind(0);
    drawFullscreen();

    target->unbind();
}

Backend::Texture* PostProcessor::renderBloomBlurPass(Backend::Texture* src,
                                                     const BloomConfig& bloom) {
    Backend::Texture* source = src;
    const int iterations = std::max(0, bloom.iterations);
    bool horizontal = true;

    _blurShader->use();
    _blurShader->setInt("uImage", 0);
    for (int i = 0; i < iterations; ++i) {
        Backend::Framebuffer* target =
            _bloomPingPongFBO[horizontal ? 1 : 0].get();
        target->bind();
        _device->setViewport(0, 0, _bloomWidth, _bloomHeight);
        _device->clear(0.f, 0.f, 0.f, 1.f);
        _blurShader->setBool("uHorizontal", horizontal);
        source->bind(0);
        drawFullscreen();
        target->unbind();

        source = target->getColorTexture();
        horizontal = !horizontal;
    }

    return source;
}

void PostProcessor::renderBloomCompositePass(Backend::Texture* scene,
                                             Backend::Texture* bloom,
                                             Backend::Framebuffer* target,
                                             float intensity) {
    target->bind();
    _device->setViewport(0, 0, _width, _height);
    _device->clear(0.f, 0.f, 0.f, 1.f);
    _device->setDepthTest(false);

    _bloomCompositeShader->use();
    _bloomCompositeShader->setInt("uScene", 0);
    _bloomCompositeShader->setInt("uBloom", 1);
    _bloomCompositeShader->setFloat("uBloomIntensity", intensity);
    scene->bind(0);
    bloom->bind(1);
    drawFullscreen();

    target->unbind();
}

void PostProcessor::renderToneMapPass(Backend::Texture* src, float gamma,
                                      ToneMapMode toneMapMode,
                                      float tonemapExposure) {
    _outputFBO->bind();
    _device->setViewport(0, 0, _width, _height);
    _device->clear(0.f, 0.f, 0.f, 1.f);
    _device->setDepthTest(false);

    _toneMapShader->use();
    _toneMapShader->setInt("uScreen", 0);
    _toneMapShader->setFloat("uGamma", gamma < 0.01f ? 1.f : gamma);
    _toneMapShader->setInt("uToneMapMode", static_cast<int>(toneMapMode));
    _toneMapShader->setFloat("uToneMapExposure", tonemapExposure);
    src->bind(0);
    drawFullscreen();

    _device->setDepthTest(true);
    _outputFBO->unbind();
}

Backend::Texture* PostProcessor::getResult() {
    return _outputFBO->getColorTexture();
}

Backend::Framebuffer* PostProcessor::getOutputFramebuffer() {
    return _outputFBO.get();
}

void PostProcessor::blitToScreen(int width, int height) {
    _outputFBO->blitToScreen(width, height);
}

void PostProcessor::resize(int width, int height) {
    _width = width;
    _height = height;
    _outputFBO->resize(width, height);
    if (_bloomCompositeFBO)
        _bloomCompositeFBO->resize(width, height);
    if (_bloomPingPongFBO[0] && _bloomPingPongFBO[1]) {
        const int width = std::max(1, _width / std::max(1, _bloomDownsample));
        const int height = std::max(1, _height / std::max(1, _bloomDownsample));
        _bloomPingPongFBO[0]->resize(width, height);
        _bloomPingPongFBO[1]->resize(width, height);
        _bloomWidth = width;
        _bloomHeight = height;
    }
}

} // namespace KE
