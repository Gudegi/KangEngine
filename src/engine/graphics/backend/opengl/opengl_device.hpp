///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _OPENGL_DEVICE_HPP_
#define _OPENGL_DEVICE_HPP_

#include <cstddef>
#include <cstdint>
#include <glad/glad.h>
#include <memory>
#include <thread>
#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
#include <cuda_gl_interop.h>
#endif
#include "../base/graphics_device.hpp"
#include "utils/types.hpp"

namespace KE {
namespace Backend {

class OpenGLBuffer : public Buffer {
  private:
    GLuint _buffer;
    GLenum _target;
    BufferType _type;
    BufferUsage _usage = BufferUsage::None;
    size_t _size;
#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
    cudaGraphicsResource* _cudaResource = nullptr;
#endif

  public:
    OpenGLBuffer(BufferType type, size_t size, const void* data = nullptr);
    OpenGLBuffer(const BufferDesc& desc, const void* data = nullptr);
    ~OpenGLBuffer() override;

    void bind() override;
    void unbind() override;
    void setData(const void* data, size_t size, size_t offset = 0) override;
#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
    bool setExternalData(const Sim::GpuArrayView& view, size_t count,
                         size_t elementSize, size_t sourceStrideBytes) override;
    cudaGraphicsResource* cudaResource();
#endif
    BufferType getType() const override { return _type; }
    BufferUsage getUsage() const override { return _usage; }
    size_t getSize() const override { return _size; }
    GLuint getHandle() const { return _buffer; }
    size_t size() const { return _size; }
};

class OpenGLShader : public Shader {
  private:
    GLuint _shaderProgram;
    ShaderDesc _desc;

    std::string loadFile(const std::string& path);
    GLuint compile(const std::string& source, GLenum type);
    GLuint link(GLuint vertexShader, GLuint fragmentShader);
    void checkCompileError(GLuint shader);
    void checkLinkError(GLuint shaderProgram);

  public:
    OpenGLShader(const ShaderDesc& desc);
    ~OpenGLShader() override;

    const std::string& getName() const override { return _desc.name; }
    const ShaderDesc& getDesc() const override { return _desc; }

    void bind() override;
    void unbind() override;

    // KE::Shader compatibility
    void use() override;

    // Uniform setters - KE::Shader compatible
    void setBool(const std::string& name, bool value) override;
    void setInt(const std::string& name, int value) override;
    void setFloat(const std::string& name, float value) override;
    void setColor(const std::string& name, float r, float g, float b,
                  float a) override;

    void setVec2(const std::string& name, const glm::vec2& value) override;
    void setVec2(const std::string& name, float x, float y) override;
    void setVec3(const std::string& name, const glm::vec3& value) override;
    void setVec3(const std::string& name, float x, float y, float z) override;
    void setVec4(const std::string& name, const glm::vec4& value) override;
    void setVec4(const std::string& name, float x, float y, float z,
                 float w) override;
    void setMat2(const std::string& name, const glm::mat2& value) override;
    void setMat3(const std::string& name, const glm::mat3& value) override;
    void setMat4(const std::string& name, const glm::mat4& value) override;
    void setMat4Array(const std::string& name, const glm::mat4* values,
                      size_t count) override;
    void setUniformBlockBinding(const std::string& blockName,
                                int slot) override;
    GLuint getHandle() const { return _shaderProgram; }
};

class OpenGLTexture : public Texture {
  private:
    GLuint _textureID;
    GLenum _target = GL_TEXTURE_2D;
    int _width, _height, _channels;
    TextureFormat _format = TextureFormat::Undefined;
    TextureUsage _usage = TextureUsage::None;
    uint32_t _mipLevelCount = 1;
    uint32_t _sampleCount = 1;
    uint32_t _depthOrArrayLayers = 1;
    TextureDimension _dimension = TextureDimension::D2;
    bool _portableResource = false;
    float _warpParam, _filterMinParam, _filterMaxParam;

  public:
    OpenGLTexture(const TextureDesc& desc);
    OpenGLTexture(const TextureDesc& desc, const SamplerDesc& sampler);
    OpenGLTexture(const TextureDesc& desc, float warpParam,
                  float filterMinParam, float filterMaxParam);
    OpenGLTexture(const TextureResourceDesc& desc,
                  const TextureInitialData* initialData);
    OpenGLTexture(GLuint cubemapHandle, int faceWidth, int faceHeight);
    // Empty color texture for FBO color attachment.
    OpenGLTexture(int w, int h, FramebufferColorFormat colorFormat);
    // Depth (or depth+stencil) texture for FBO attachment
    OpenGLTexture(int w, int h, bool stencil);
    ~OpenGLTexture() override;

    void bind(int slot = 0) override;
    void unbind() override;
    void setWrapParam(GLenum wrapU, GLenum wrapV) const;
    void setWarpParam(GLfloat warpParam = GL_REPEAT) const;
    void setFilterParam(GLfloat filterMinParam = GL_LINEAR_MIPMAP_LINEAR,
                        GLfloat filterMaxParam = GL_LINEAR) const;
    int getWidth() const override { return _width; }
    int getHeight() const override { return _height; }
    TextureFormat getFormat() const override { return _format; }
    TextureUsage getUsage() const override { return _usage; }
    uint32_t getMipLevelCount() const override { return _mipLevelCount; }
    uint32_t getSampleCount() const override { return _sampleCount; }
    uint32_t getDepthOrArrayLayers() const override {
        return _depthOrArrayLayers;
    }
    TextureDimension getDimension() const override { return _dimension; }
    GLenum getTarget() const { return _target; }
    void setSize(int w, int h) {
        _width = w;
        _height = h;
    }
    GLuint getHandle() const { return _textureID; }
    TextureFormat format() const { return _format; }
    TextureUsage usage() const { return _usage; }
    uint32_t mipLevelCount() const { return _mipLevelCount; }
    uint32_t sampleCount() const { return _sampleCount; }
    bool isPortableResource() const { return _portableResource; }
    uintptr_t getNativeHandle() const override {
        return static_cast<uintptr_t>(_textureID);
    }
};

class OpenGLTextureView : public TextureView {
  private:
    OpenGLTexture* _texture = nullptr;
    TextureViewDesc _desc;

  public:
    OpenGLTextureView(OpenGLTexture* texture, TextureViewDesc desc);
    Texture* getTexture() const override { return _texture; }
    const TextureViewDesc& getDesc() const override { return _desc; }
    GLuint getHandle() const { return _texture->getHandle(); }
    GLenum getTarget() const { return _texture->getTarget(); }
};

class OpenGLSampler : public Sampler {
  private:
    GLuint _sampler = 0;

  public:
    explicit OpenGLSampler(const SamplerDesc& desc);
    ~OpenGLSampler() override;
    GLuint getHandle() const { return _sampler; }
};

class OpenGLRenderTarget : public RenderTarget {
  private:
    GLuint _fbo = 0;
    GLuint _resolveFbo = 0;
    RenderPassDesc _desc;
    int _width = 0;
    int _height = 0;
    uint32_t _sampleCount = 1;
    std::vector<GLenum> _drawBuffers;
    std::vector<OpenGLTexture*> _resolveTextures;

  public:
    explicit OpenGLRenderTarget(const RenderPassDesc& desc);
    ~OpenGLRenderTarget() override;

    const RenderPassDesc& getDesc() const override { return _desc; }
    int getWidth() const override { return _width; }
    int getHeight() const override { return _height; }
    uint32_t getSampleCount() const override { return _sampleCount; }
    GLuint getHandle() const { return _fbo; }

    // Backend execution hooks. OpenGLDevice::submit owns thread validation.
    void beginPass();
    void endPass();
};

class OpenGLBindGroupLayout : public BindGroupLayout {
    BindGroupLayoutDesc _desc;
  public:
    explicit OpenGLBindGroupLayout(BindGroupLayoutDesc desc);
    const BindGroupLayoutDesc& getDesc() const override { return _desc; }
};

class OpenGLPipelineLayout : public PipelineLayout {
    PipelineLayoutDesc _desc;
  public:
    explicit OpenGLPipelineLayout(PipelineLayoutDesc desc);
    const PipelineLayoutDesc& getDesc() const override { return _desc; }
};

class OpenGLBindGroup : public BindGroup {
    BindGroupDesc _desc;
  public:
    explicit OpenGLBindGroup(BindGroupDesc desc);
    const BindGroupDesc& getDesc() const override { return _desc; }
};

class OpenGLGraphicsPipeline : public GraphicsPipeline {
  private:
    GraphicsPipelineDesc _desc;
    std::unique_ptr<OpenGLShader> _shader;
    GLuint _vao = 0;
    struct CachedBinding {
        uint32_t group = 0;
        uint32_t binding = 0;
        BindingType type = BindingType::UniformBuffer;
        GLint location = -1;
        GLuint slot = 0;
    };
    std::vector<CachedBinding> _bindings;

  public:
    explicit OpenGLGraphicsPipeline(const GraphicsPipelineDesc& desc);
    ~OpenGLGraphicsPipeline() override;
    const GraphicsPipelineDesc& getDesc() const override { return _desc; }
    void apply() const;
    void bindVertexBuffer(uint32_t slot, OpenGLBuffer* buffer,
                          uint64_t offset) const;
    void bindIndexBuffer(OpenGLBuffer* buffer) const;
    void bindGroup(uint32_t index, OpenGLBindGroup* group) const;
    std::vector<GLuint> textureSlots() const;
    std::vector<GLuint> uniformSlots() const;
};

class OpenGLVertexArray : public VertexArray {
  private:
    GLuint _vao;

  public:
    OpenGLVertexArray();
    ~OpenGLVertexArray() override;

    void bind() override;
    void unbind() override;
    void setVertexAttribute(const VertexAttribute& attribute) override;
    void setVertexBuffer(Buffer* buffer) override;
    void setIndexBuffer(Buffer* buffer) override;
};

class OpenGLFramebuffer : public Framebuffer {
  private:
    // --- Texture FBO (non-MSAA scene FBO) ---
    GLuint _fbo = 0;
    std::unique_ptr<OpenGLTexture> _colorTexObj;
    std::unique_ptr<OpenGLTexture>
        _depthTexObj; // depth (or depth+stencil) texture

    // --- [SIMPLE RBO] non-MSAA depth+stencil renderbuffer ---
    // GLuint _rbo = 0; // GL_DEPTH24_STENCIL8
    // attach: glFramebufferRenderbuffer(GL_DEPTH_STENCIL_ATTACHMENT, _rbo)
    // faster than texture when depth sampling not needed

    // --- MSAA FBO (RBO-based) ---
    GLuint _msaaFbo = 0;
    GLuint _msaaColorRbo = 0;
    GLuint _msaaDepthRbo = 0; // GL_DEPTH_COMPONENT32

    FramebufferDesc _desc;
    // TODO: Add ping-pong PBO readback
    std::unique_ptr<OpenGLFramebuffer> _scaledReadbackFramebuffer;

  public:
    OpenGLFramebuffer(const FramebufferDesc& desc);
    ~OpenGLFramebuffer() override;

    void bind() override;
    void unbind() override;
    void resize(int scrWidth, int scrHeight) override;
    Texture* getColorTexture() override;
    Texture* getDepthTexture() override;
    Texture* getStencilTexture() override;
    Texture* getDepthStencilTexture() override;

    // MSAA resolve: blit _msaaFbo -> _fbo (If msaaSamples == 0, no-op)
    void resolve() override;
    // Final blit: _fbo -> default framebuffer (screen)
    void blitToScreen(int scrWidth, int scrHeight) override;
    std::vector<uint8_t> readColorPixels(bool flipY = true) override;
    std::vector<uint8_t> readColorPixelsResized(int width, int height,
                                                bool flipY = true) override;
};

class OpenGLDevice : public GraphicsDevice {
  private:
    bool _initialized;
    bool _validationEnabled = false;
    std::thread::id _renderThread;

    // Skybox
    GLuint _skyboxVAO = 0;
    GLuint _skyboxTex = 0;
    std::unique_ptr<Shader> _skyboxShader;
    UpAxis _skyboxUpAxis = UpAxis::Y;
    // 6 individual face images: +X, -X, +Y, -Y, +Z, -Z(OpenGL Y up frame)
    GLuint loadCubemap(const std::vector<std::string>& paths);
    // Single cross-layout image (horizontal 4:3 or vertical 3:4)
    GLuint loadCubemapCross(const std::string& path);

    GLuint makeSkyboxVAO();
    void applySkyboxTex(GLuint tex, UpAxis upAxis);

  public:
    OpenGLDevice();
    ~OpenGLDevice() override;

    void initialize() override;
    void shutdown() override;
    BackendType getBackendType() const override { return BackendType::OpenGL; }
    void setValidationEnabled(bool enabled) override {
        _validationEnabled = enabled;
    }
    bool isValidationEnabled() const override { return _validationEnabled; }

    // Rendering
    void beginFrame() override;
    void endFrame() override;
    void clear(float r, float g, float b, float a) override;
    void setViewport(int x, int y, int width, int height) override;
    void drawIndexed(size_t indexCount) override;
    void drawLines(size_t vertexCount) override;
    void drawPoints(size_t vertexCount) override;
    void drawIndexedInstanced(size_t indexCount, size_t instanceCount) override;
    void checkError() override;

    // Render State
    void setDepthTest(bool enable) override;
    void setDepthWrite(bool enable) override;
    void setColorWrite(bool enable) override;
    void setBlend(bool enable) override;
    void setBlendFunc(BlendFactor src, BlendFactor dst) override;
    void setStencilTest(bool enable) override;
    void setStencilFunc(StencilFunc func, int ref, uint32_t mask) override;
    void setStencilOp(StencilOp stencilFail, StencilOp depthFail,
                      StencilOp depthPass) override;
    void setStencilWriteMask(uint32_t mask) override;
    void setPolygonMode(PolygonMode mode) override;
    void setLineWidth(float width) override;
    void setCullFace(bool enable) override;
    void setCullFaceMode(CullFaceMode mode) override;
    void setClearColor(float r, float g, float b, float a) override;

    // Resource creation
    std::unique_ptr<Buffer> createBuffer(BufferType type, size_t size,
                                         const void* data = nullptr) override;
    std::unique_ptr<Buffer>
    createBuffer(const BufferDesc& desc, const void* data = nullptr) override;
    std::unique_ptr<Shader> createShader(const ShaderDesc& desc) override;
    std::unique_ptr<Texture> createTexture(const TextureDesc& desc) override;
    std::unique_ptr<Texture> createTexture(const TextureDesc& desc,
                                           const SamplerDesc& sampler) override;
    std::unique_ptr<Texture>
    createTexture(const TextureResourceDesc& desc,
                  const TextureInitialData* initialData = nullptr) override;
    std::unique_ptr<Texture>
    createCubemapTexture(const std::string& crossPath) override;
    std::unique_ptr<Texture> createCubemapTexture(
        const std::vector<std::string>& facePaths) override;
    std::unique_ptr<TextureView>
    createTextureView(Texture* texture,
                      const TextureViewDesc& desc = {}) override;
    std::unique_ptr<Sampler>
    createSampler(const SamplerDesc& desc = {}) override;
    std::unique_ptr<RenderTarget>
    createRenderTarget(const RenderPassDesc& desc) override;
    void beginLegacyRenderPass(RenderTarget* target) override;
    void endLegacyRenderPass(RenderTarget* target) override;
    std::unique_ptr<GraphicsPipeline>
    createGraphicsPipeline(const GraphicsPipelineDesc& desc) override;
    std::unique_ptr<BindGroupLayout>
    createBindGroupLayout(const BindGroupLayoutDesc& desc) override;
    std::unique_ptr<PipelineLayout>
    createPipelineLayout(const PipelineLayoutDesc& desc) override;
    std::unique_ptr<BindGroup>
    createBindGroup(const BindGroupDesc& desc) override;
    TextureReadback readTexture(TextureView* view) override;
    std::unique_ptr<CommandEncoder> createCommandEncoder() override;
    void submit(CommandBuffer& commandBuffer) override;
    std::unique_ptr<VertexArray> createVertexArray() override;

    // Convenience shader creation methods (KE::Shader compatible)
    std::unique_ptr<Shader> createShader(const char* vertexSource,
                                         const char* fragmentSource) override;
    std::unique_ptr<Shader>
    createShader(const std::string& vertexSource,
                 const std::string& fragmentSource) override;
    std::unique_ptr<Texture> createTexture(const std::string path,
                                           bool flip = false) override;
    std::unique_ptr<Texture> createTexture(const std::string path, bool flip,
                                           const SamplerDesc& sampler) override;
    std::unique_ptr<Texture>
    createTexture(const std::string path, bool flip = false,
                  float warpParam = GL_REPEAT,
                  float minFilferParam = GL_LINEAR_MIPMAP_LINEAR,
                  float maxFilterParam = GL_LINEAR) override;
    std::unique_ptr<Framebuffer>
    createFramebuffer(const FramebufferDesc& desc) override;
#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
    bool mapCudaBuffers(const std::vector<Buffer*>& buffers,
                        std::vector<Sim::GpuArrayView>& views, size_t count,
                        size_t elementSize, int deviceId,
                        uint64_t streamHandle) override;
    void unmapCudaBuffers(const std::vector<Buffer*>& buffers, int deviceId,
                          uint64_t streamHandle) override;
#endif

    void bindUniformBuffer(Buffer* buffer, int slot) override;

    // Skybox
    void setSkybox(const std::string& path, UpAxis upAxis = UpAxis::Y) override;
    void setSkybox(const std::vector<std::string>& paths,
                   UpAxis upAxis = UpAxis::Y) override;
    void drawSkybox(const glm::mat4& view, const glm::mat4& proj) override;
};

} // namespace Backend
} // namespace KE

#endif
