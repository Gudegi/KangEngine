#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/opengl/opengl_device.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace KE;
using namespace KE::Backend;

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void expectInternalFormat(Texture* texture, GLint expected) {
    auto* glTexture = dynamic_cast<OpenGLTexture*>(texture);
    require(glTexture != nullptr, "texture is not an OpenGL texture");
    glBindTexture(GL_TEXTURE_2D, glTexture->getHandle());
    GLint actual = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT,
                             &actual);
    glBindTexture(GL_TEXTURE_2D, 0);
    require(actual == expected, "OpenGL texture internal format mismatch");
}

TextureResourceDesc textureDesc(TextureFormat format, TextureUsage usage,
                                const char* label) {
    TextureResourceDesc desc;
    desc.extent = {4, 4, 1};
    desc.format = format;
    desc.usage = usage;
    desc.label = label;
    return desc;
}

std::unique_ptr<Shader> createSamplingShader(OpenGLDevice& device) {
    constexpr const char* vertexSource = R"(
#version 410 core
out vec2 vUV;
void main() {
    const vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    vec2 position = positions[gl_VertexID];
    vUV = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
)";
    constexpr const char* fragmentSource = R"(
#version 410 core
in vec2 vUV;
uniform sampler2D uTexture;
out vec4 outColor;
void main() { outColor = texture(uTexture, vUV); }
)";
    return device.createShader(vertexSource, fragmentSource);
}

void expectSample(OpenGLTexture* texture, OpenGLSampler* sampler,
                  Shader* shader, GLuint vao,
                  const std::array<uint8_t, 4>& expected, int tolerance = 2) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, 64, 64);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_FRAMEBUFFER_SRGB);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    shader->use();
    shader->setInt("uTexture", 0);
    texture->bind(0);
    glBindSampler(0, sampler->getHandle());
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    std::array<uint8_t, 4> pixel{};
    glReadBuffer(GL_BACK);
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel.data());
    for (size_t channel = 0; channel < pixel.size(); ++channel) {
        const int delta = std::abs(static_cast<int>(pixel[channel]) -
                                   static_cast<int>(expected[channel]));
        require(delta <= tolerance, "sampled texture pixel mismatch");
    }
    glBindVertexArray(0);
    glBindSampler(0, 0);
    texture->unbind();
}

} // namespace

int main() {
    Window window;
    window.init(64, 64, true);
    require(window.getGlfwWindow() != nullptr,
            "failed to create the hidden OpenGL smoke window");

    OpenGLDevice device;
    device.initialize();

    constexpr TextureUsage sampledAttachment =
        TextureUsage::TextureBinding | TextureUsage::RenderAttachment;
    std::array<uint8_t, 4 * 4 * 4> rgbaData{};
    for (size_t i = 0; i < rgbaData.size(); i += 4) {
        rgbaData[i + 0] = 32;
        rgbaData[i + 1] = 64;
        rgbaData[i + 2] = 128;
        rgbaData[i + 3] = 255;
    }
    TextureInitialData initialRgba{rgbaData.data(), rgbaData.size(), 0};

    auto rgba8Desc = textureDesc(TextureFormat::RGBA8Unorm,
                                 sampledAttachment | TextureUsage::CopyDst,
                                 "rhi_smoke_rgba8");
    auto rgba8 = device.createTexture(rgba8Desc, &initialRgba);
    expectInternalFormat(rgba8.get(), GL_RGBA8);
    TextureViewDesc rgba8ViewDesc;
    rgba8ViewDesc.format = TextureFormat::RGBA8Unorm;
    rgba8ViewDesc.label = "rhi_smoke_rgba8_view";
    auto rgba8View = device.createTextureView(rgba8.get(), rgba8ViewDesc);
    require(rgba8View->getTexture() == rgba8.get(),
            "texture view references the wrong texture");

    auto srgbDesc = textureDesc(TextureFormat::RGBA8UnormSrgb,
                                sampledAttachment | TextureUsage::CopyDst,
                                "rhi_smoke_srgb");
    auto srgb = device.createTexture(srgbDesc, &initialRgba);
    expectInternalFormat(srgb.get(), GL_SRGB8_ALPHA8);

    auto rgba16f = device.createTexture(
        textureDesc(TextureFormat::RGBA16Float, sampledAttachment,
                    "rhi_smoke_rgba16f"));
    expectInternalFormat(rgba16f.get(), GL_RGBA16F);

    std::array<uint8_t, 4 * 4> redData{};
    TextureInitialData initialRed{redData.data(), redData.size(), 0};
    auto r8 = device.createTexture(
        textureDesc(TextureFormat::R8Unorm,
                    TextureUsage::TextureBinding | TextureUsage::CopyDst,
                    "rhi_smoke_r8"),
        &initialRed);
    expectInternalFormat(r8.get(), GL_R8);

    auto depth32 = device.createTexture(textureDesc(
        TextureFormat::Depth32Float, TextureUsage::RenderAttachment,
        "rhi_smoke_depth32"));
    expectInternalFormat(depth32.get(), GL_DEPTH_COMPONENT32F);
    TextureViewDesc depthViewDesc;
    depthViewDesc.format = TextureFormat::Depth32Float;
    depthViewDesc.aspect = TextureAspect::DepthOnly;
    depthViewDesc.label = "rhi_smoke_depth_view";
    auto depthView =
        device.createTextureView(depth32.get(), depthViewDesc);
    require(depthView->getTexture() == depth32.get(),
            "depth view references the wrong texture");

    auto depthStencil = device.createTexture(textureDesc(
        TextureFormat::Depth24Stencil8, TextureUsage::RenderAttachment,
        "rhi_smoke_depth_stencil"));
    expectInternalFormat(depthStencil.get(), GL_DEPTH24_STENCIL8);

    SamplerDesc samplerDesc;
    samplerDesc.wrapU = TextureWrap::ClampToEdge;
    samplerDesc.wrapV = TextureWrap::MirroredRepeat;
    samplerDesc.minFilter = TextureFilter::Nearest;
    samplerDesc.magFilter = TextureFilter::Linear;
    samplerDesc.label = "rhi_smoke_sampler";
    auto sampler = device.createSampler(samplerDesc);
    auto* glSampler = dynamic_cast<OpenGLSampler*>(sampler.get());
    require(glSampler != nullptr && glSampler->getHandle() != 0,
            "OpenGL sampler creation failed");

    auto samplingShader = createSamplingShader(device);
    GLuint samplingVao = 0;
    glGenVertexArrays(1, &samplingVao);
    expectSample(dynamic_cast<OpenGLTexture*>(rgba8.get()), glSampler,
                 samplingShader.get(), samplingVao, {32, 64, 128, 255});
    // Sampling an sRGB texture decodes RGB into linear values. The default
    // framebuffer is intentionally left non-sRGB for this observable check.
    expectSample(dynamic_cast<OpenGLTexture*>(srgb.get()), glSampler,
                 samplingShader.get(), samplingVao, {4, 13, 55, 255}, 3);

    auto* glRgba16f = dynamic_cast<OpenGLTexture*>(rgba16f.get());
    require(glRgba16f != nullptr, "RGBA16F texture is not OpenGL-backed");
    GLuint colorFbo = 0;
    glGenFramebuffers(1, &colorFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, colorFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           glRgba16f->getHandle(), 0);
    require(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,
            "RGBA16F render attachment is incomplete");
    glViewport(0, 0, 4, 4);
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    expectSample(glRgba16f, glSampler, samplingShader.get(), samplingVao,
                 {64, 128, 191, 255}, 3);

    auto* glDepthStencil =
        dynamic_cast<OpenGLTexture*>(depthStencil.get());
    require(glDepthStencil != nullptr,
            "depth-stencil texture is not OpenGL-backed");
    GLuint depthFbo = 0;
    glGenFramebuffers(1, &depthFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, depthFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                           GL_TEXTURE_2D, glDepthStencil->getHandle(), 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    require(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,
            "depth-only render attachment is incomplete");
    glClearDepth(0.375);
    glClearStencil(3);
    glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    float depthPixel = 0.0f;
    glReadPixels(2, 2, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depthPixel);
    require(std::abs(depthPixel - 0.375f) < 0.001f,
            "depth attachment clear/readback mismatch");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    GLuint invalidColorFbo = 0;
    glGenFramebuffers(1, &invalidColorFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, invalidColorFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           glDepthStencil->getHandle(), 0);
    require(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE,
            "depth texture was accepted as an ordinary color attachment");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    bool rejectedSmallUpload = false;
    try {
        TextureInitialData tooSmall{rgbaData.data(), 4, 0};
        auto invalid = device.createTexture(rgba8Desc, &tooSmall);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejectedSmallUpload = true;
    }
    require(rejectedSmallUpload, "undersized texture upload was accepted");

    bool rejectedColorDepthView = false;
    try {
        TextureViewDesc invalidView;
        invalidView.aspect = TextureAspect::DepthOnly;
        auto invalid = device.createTextureView(rgba8.get(), invalidView);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejectedColorDepthView = true;
    }
    require(rejectedColorDepthView,
            "color texture accepted a depth-only texture view");

    // Cubemap resource/view contract used by the RHI skybox pipeline.
    std::array<std::string, 6> cubeFacePaths{};
    for (size_t face = 0; face < cubeFacePaths.size(); ++face) {
        cubeFacePaths[face] = "/tmp/kang_rhi_cube_face_" +
                              std::to_string(face) + ".ppm";
        std::ofstream file(cubeFacePaths[face], std::ios::binary);
        file << "P6\n1 1\n255\n";
        const std::array<unsigned char, 3> pixel{
            static_cast<unsigned char>(32 + face * 16), 64, 128};
        file.write(reinterpret_cast<const char*>(pixel.data()), pixel.size());
    }
    const std::vector<std::string> cubeFaces(cubeFacePaths.begin(),
                                              cubeFacePaths.end());
    auto cube = device.createCubemapTexture(cubeFaces);
    auto* glCube = dynamic_cast<OpenGLTexture*>(cube.get());
    require(glCube && glCube->getTarget() == GL_TEXTURE_CUBE_MAP &&
                glCube->getDepthOrArrayLayers() == 6,
            "cubemap texture metadata mismatch");
    TextureViewDesc cubeViewDesc;
    cubeViewDesc.dimension = TextureViewDimension::Cube;
    cubeViewDesc.arrayLayerCount = 6;
    cubeViewDesc.label = "rhi_smoke_cube_view";
    auto cubeView = device.createTextureView(cube.get(), cubeViewDesc);
    BindGroupLayoutDesc cubeLayoutDesc;
    cubeLayoutDesc.entries = {
        {0, BindingType::SampledTexture, ShaderStageVisibility::Fragment,
         TextureFormat::Undefined, TextureSampleType::Float,
         TextureViewDimension::Cube},
        {1, BindingType::Sampler, ShaderStageVisibility::Fragment}};
    auto cubeLayout = device.createBindGroupLayout(cubeLayoutDesc);
    BindGroupDesc cubeGroupDesc;
    cubeGroupDesc.layout = cubeLayout.get();
    cubeGroupDesc.entries = {
        {0, nullptr, 0, 0, cubeView.get(), nullptr},
        {1, nullptr, 0, 0, nullptr, sampler.get()}};
    auto cubeGroup = device.createBindGroup(cubeGroupDesc);
    require(cubeGroup != nullptr, "cubemap bind group creation failed");
    for (const std::string& path : cubeFacePaths)
        std::remove(path.c_str());

    for (int iteration = 0; iteration < 32; ++iteration) {
        GLuint releasedHandle = 0;
        {
            auto temporary = device.createTexture(textureDesc(
                TextureFormat::RGBA8Unorm, sampledAttachment,
                "rhi_smoke_recreate"));
            auto* glTemporary = dynamic_cast<OpenGLTexture*>(temporary.get());
            require(glTemporary != nullptr, "temporary texture creation failed");
            releasedHandle = glTemporary->getHandle();
            require(glIsTexture(releasedHandle) == GL_TRUE,
                    "created texture handle is not live");
        }
        require(glIsTexture(releasedHandle) == GL_FALSE,
                "destroyed texture handle remains live");
    }

    require(glGetError() == GL_NO_ERROR, "OpenGL error in resource smoke");
    glDeleteFramebuffers(1, &invalidColorFbo);
    glDeleteFramebuffers(1, &depthFbo);
    glDeleteFramebuffers(1, &colorFbo);
    glDeleteVertexArrays(1, &samplingVao);
    std::cout << "PASS: OpenGL RHI textures, cube views, formats, and sampler"
              << std::endl;
    return 0;
}
