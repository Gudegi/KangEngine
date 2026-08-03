///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _GRAPHICS_DEVICE_HPP_
#define _GRAPHICS_DEVICE_HPP_

#include "rhi_types.hpp"
#include "shader_preprocessor.hpp"
#include "sim/gpu_array_view.hpp"
#include "utils/types.hpp"
#include <glm/glm.hpp>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace KE {
namespace Backend {

using std::string;
using std::unique_ptr;
using std::vector;

enum class BackendType { OpenGL, Vulkan, WebGPU };

// Legacy allocation categories used by the immediate OpenGL renderer. New RHI
// resources should describe combinable intent through BufferUsage/BufferDesc.
enum class BufferType { Vertex, DynamicVertex, Index, Uniform };

enum class ShaderType { Vertex, Fragment, Geometry, Compute };

enum class UniformType { Int, Float, Vec2, Vec3, Vec4, Mat3, Mat4 };

enum class VertexAttributeType { Float, Int, UnsignedInt, Byte, UnsignedByte };

enum class PolygonMode { Fill, Line, Point };

enum class CullFaceMode { Front, Back };

enum class StencilFunc {
    Never,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    Always,
};

enum class StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
};

enum class BlendFactor {
    Zero,
    One,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
};

enum class FramebufferColorFormat {
    RGBA8,
    RGBA16F,
};

enum class TextureWrap {
    Repeat,
    ClampToEdge,
    MirroredRepeat,
};

enum class TextureFilter {
    Nearest,
    Linear,
    LinearMipmapLinear,
};

struct VertexAttribute {
    int location;
    int size;
    VertexAttributeType type;
    bool normalized;
    size_t stride;
    size_t offset;
    int divisor = 0; // 0=per-vertex, 1=per-instance
};

struct ShaderStage {
    std::string source;
    ShaderType type;
    std::string entryPoint = "main";
};

struct ShaderDesc {
    std::vector<ShaderStage> stages;
    std::string name;
};

class Buffer;
class TextureView;
class Sampler;
class BindGroupLayout;
class PipelineLayout;

struct PipelineLayoutDesc {
    std::vector<BindGroupLayout*> bindGroupLayouts;
    std::string label;
};

struct BindGroupEntry {
    uint32_t binding = 0;
    Buffer* buffer = nullptr;
    uint64_t offset = 0;
    uint64_t size = 0;
    TextureView* textureView = nullptr;
    Sampler* sampler = nullptr;
};

struct BindGroupDesc {
    BindGroupLayout* layout = nullptr;
    std::vector<BindGroupEntry> entries;
    std::string label;
};

struct GraphicsPipelineDesc {
    ShaderDesc shader;
    std::vector<VertexBufferLayout> vertexBuffers;
    PrimitiveState primitive;
    RasterState raster;
    std::optional<DepthStencilState> depthStencil;
    std::vector<ColorTargetState> colorTargets;
    uint32_t sampleCount = 1;
    // WP6 replaces this placeholder with a typed PipelineLayout reference.
    PipelineLayout* pipelineLayout = nullptr;
    std::string label;
};

// Legacy image-upload descriptor. New portable allocations use
// TextureResourceDesc with explicit format and usage.
struct TextureDesc {
    int width, height;
    int channels = 4;
    const void* data = nullptr;
    std::string name;
};

struct SamplerDesc {
    TextureWrap wrapU = TextureWrap::Repeat;
    TextureWrap wrapV = TextureWrap::Repeat;
    TextureFilter minFilter = TextureFilter::LinearMipmapLinear;
    TextureFilter magFilter = TextureFilter::Linear;
    std::string label;
};

// Legacy single-color framebuffer descriptor. Do not extend this for MRT; new
// rendering uses explicit color/depth attachment descriptions.
struct FramebufferDesc {
    int width, height;
    bool depthOnly = false; // shadow FBO: no color attachment
    bool stencil = false;   // use depth+stencil
    int msaaSamples = 0;    // 0 means No MSAA
    FramebufferColorFormat colorFormat = FramebufferColorFormat::RGBA8;
};

// Forward declarations
class Buffer;
class Shader;
class Texture;
class TextureView;
class Sampler;
class RenderTarget;
class GraphicsPipeline;
class BindGroup;
class CommandEncoder;
class CommandBuffer;
class VertexArray;
class Framebuffer;

class GraphicsDevice {
  public:
    virtual ~GraphicsDevice() = default;

    virtual void initialize() = 0;
    virtual void shutdown() = 0;
    virtual BackendType getBackendType() const = 0;
    virtual void setValidationEnabled(bool) {}
    virtual bool isValidationEnabled() const { return false; }

    // Rendering
    virtual void beginFrame() = 0;
    virtual void endFrame() = 0;
    virtual void clear(float r, float g, float b, float a) = 0;
    virtual void setViewport(int x, int y, int width, int height) = 0;
    virtual void drawIndexed(size_t indexCount) = 0;
    virtual void drawLines(size_t vertexCount) = 0;
    virtual void drawPoints(size_t vertexCount) = 0;
    // instanced rendering
    virtual void drawIndexedInstanced(size_t indexCount,
                                      size_t instanceCount) = 0;
    virtual void checkError() = 0;

    // Render State
    virtual void setDepthTest(bool enable) = 0;
    virtual void setDepthWrite(bool enable) = 0;
    virtual void setColorWrite(bool enable) = 0;
    virtual void setBlend(bool enable) = 0;
    virtual void setBlendFunc(BlendFactor src, BlendFactor dst) = 0;
    virtual void setStencilTest(bool enable) = 0;
    virtual void setStencilFunc(StencilFunc func, int ref, uint32_t mask) = 0;
    virtual void setStencilOp(StencilOp stencilFail, StencilOp depthFail,
                              StencilOp depthPass) = 0;
    virtual void setStencilWriteMask(uint32_t mask) = 0;
    virtual void setPolygonMode(PolygonMode mode) = 0;
    virtual void setLineWidth(float width) = 0;
    virtual void setCullFace(bool enable) = 0;
    virtual void setCullFaceMode(CullFaceMode mode) = 0;
    virtual void setClearColor(float r, float g, float b, float a) = 0;

    // Resource creation
    virtual std::unique_ptr<Buffer>
    createBuffer(BufferType type, size_t size, const void* data = nullptr) = 0;
    virtual std::unique_ptr<Buffer>
    createBuffer(const BufferDesc& desc, const void* data = nullptr) = 0;
    virtual void bindUniformBuffer(Buffer* buffer, int slot) = 0;
    virtual std::unique_ptr<Shader> createShader(const ShaderDesc& desc) = 0;
    virtual std::unique_ptr<Texture> createTexture(const TextureDesc& desc) = 0;
    virtual std::unique_ptr<Texture> createTexture(const TextureDesc& desc,
                                                   const SamplerDesc&) {
        return createTexture(desc);
    }
    virtual std::unique_ptr<Texture>
    createTexture(const TextureResourceDesc& desc,
                  const TextureInitialData* initialData = nullptr) = 0;
    virtual std::unique_ptr<Texture>
    createCubemapTexture(const std::string& crossPath) {
        throw std::runtime_error("cubemap textures are unsupported");
    }
    virtual std::unique_ptr<Texture>
    createCubemapTexture(const std::vector<std::string>& facePaths) {
        throw std::runtime_error("cubemap textures are unsupported");
    }
    virtual std::unique_ptr<TextureView>
    createTextureView(Texture* texture, const TextureViewDesc& desc = {}) = 0;
    virtual std::unique_ptr<Sampler>
    createSampler(const SamplerDesc& desc = {}) = 0;
    virtual std::unique_ptr<RenderTarget>
    createRenderTarget(const RenderPassDesc& desc) = 0;
    // Temporary migration bridge: legacy immediate draws may target an RHI
    // render pass while production passes are converted incrementally.
    // Backends without an immediate API reject this explicitly.
    virtual void beginLegacyRenderPass(RenderTarget*) {
        throw std::runtime_error(
            "legacy rendering into an RHI target is unsupported");
    }
    virtual void endLegacyRenderPass(RenderTarget*) {
        throw std::runtime_error(
            "legacy rendering into an RHI target is unsupported");
    }
    virtual std::unique_ptr<GraphicsPipeline>
    createGraphicsPipeline(const GraphicsPipelineDesc& desc) = 0;
    virtual std::unique_ptr<BindGroupLayout>
    createBindGroupLayout(const BindGroupLayoutDesc& desc) = 0;
    virtual std::unique_ptr<PipelineLayout>
    createPipelineLayout(const PipelineLayoutDesc& desc) = 0;
    virtual std::unique_ptr<BindGroup>
    createBindGroup(const BindGroupDesc& desc) = 0;
    virtual TextureReadback readTexture(TextureView* view) = 0;
    virtual std::unique_ptr<CommandEncoder> createCommandEncoder() = 0;
    virtual void submit(CommandBuffer& commandBuffer) = 0;
    virtual std::unique_ptr<VertexArray> createVertexArray() = 0;

    // Legacy shader convenience path for OpenGL/debug shaders.
    // New backend-neutral features should use typed buffers and material/pass
    // bindings instead of expanding name-based uniforms.
    virtual std::unique_ptr<Shader>
    createShader(const char* vertexSource, const char* fragmentSource) = 0;
    virtual std::unique_ptr<Shader>
    createShader(const std::string& vertexSource,
                 const std::string& fragmentSource) = 0;

    std::unique_ptr<Shader> createShaderFromFile(const std::string& vertPath,
                                                 const std::string& fragPath) {
        ShaderDesc desc;
        desc.name = vertPath + "|" + fragPath;
        desc.stages = {
            {loadShaderSource(vertPath), ShaderType::Vertex, "main"},
            {loadShaderSource(fragPath), ShaderType::Fragment, "main"},
        };
        return createShader(desc);
    }
    virtual std::unique_ptr<Texture> createTexture(const std::string path,
                                                   bool flip = false) = 0;
    virtual std::unique_ptr<Texture>
    createTexture(const std::string path, bool flip,
                  const SamplerDesc& sampler) = 0;
    virtual std::unique_ptr<Texture> createTexture(const std::string path,
                                                   bool flip, float warpParam,
                                                   float minFilferParam,
                                                   float maxFilterParam) = 0;

    virtual std::unique_ptr<Framebuffer>
    createFramebuffer(const FramebufferDesc& desc) = 0;

    // Short-lived CUDA access to backend buffers. The buffers must be
    // unmapped before graphics commands consume them.
    virtual bool mapCudaBuffers(const std::vector<Buffer*>&,
                                std::vector<Sim::GpuArrayView>&, size_t, size_t,
                                int, uint64_t) {
        return false;
    }
    virtual void unmapCudaBuffers(const std::vector<Buffer*>&, int, uint64_t) {
        throw std::runtime_error(
            "CUDA buffer unmap is unsupported by this graphics backend");
    }

    // Skybox (optional — no-op on backends that don't support it)
    virtual void setSkybox(const std::string& path, UpAxis upAxis = UpAxis::Y) {
    }
    virtual void setSkybox(const std::vector<std::string>& paths,
                           UpAxis upAxis = UpAxis::Y) {}
    virtual void drawSkybox(const glm::mat4& view, const glm::mat4& proj) {}
};

class Buffer {
  public:
    virtual ~Buffer() = default;
    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual void setData(const void* data, size_t size, size_t offset = 0) = 0;
    // Upload external device memory without staging through CPU memory.
    // Backends return false when the source memory type is unsupported.
    virtual bool setExternalData(const Sim::GpuArrayView&, size_t, size_t,
                                 size_t) {
        return false;
    }
    virtual BufferType getType() const = 0;
    virtual BufferUsage getUsage() const = 0;
    virtual size_t getSize() const = 0;
};

// Command recording is CPU-only. Referenced resources must outlive submission.
// Backend objects are touched only by GraphicsDevice::submit on its render
// thread, leaving recording suitable for future worker/Taskflow tasks.
class RenderPassEncoder {
  public:
    virtual ~RenderPassEncoder() = default;
    virtual void setViewport(float x, float y, float width, float height,
                             float minDepth = 0.0f,
                             float maxDepth = 1.0f) = 0;
    virtual void setScissor(uint32_t x, uint32_t y, uint32_t width,
                            uint32_t height) = 0;
    // Optional wide-line state for debug rendering. Portable production
    // geometry must not depend on widths above 1; WebGPU may clamp or expand
    // debug lines into triangles.
    virtual void setLineWidth(float width) = 0;
    virtual void setPipeline(GraphicsPipeline* pipeline) = 0;
    virtual void setBindGroup(uint32_t index, BindGroup* bindGroup) = 0;
    virtual void setVertexBuffer(uint32_t slot, Buffer* buffer,
                                 uint64_t offset = 0) = 0;
    virtual void setIndexBuffer(Buffer* buffer, IndexFormat format,
                                uint64_t offset = 0) = 0;
    virtual void draw(uint32_t vertexCount, uint32_t instanceCount = 1,
                      uint32_t firstVertex = 0,
                      uint32_t firstInstance = 0) = 0;
    virtual void drawIndexed(uint32_t indexCount, uint32_t instanceCount = 1,
                             uint32_t firstIndex = 0,
                             int32_t baseVertex = 0,
                             uint32_t firstInstance = 0) = 0;
    virtual void end() = 0;
};

class CommandBuffer {
  public:
    virtual ~CommandBuffer() = default;
};

class CommandEncoder {
  public:
    virtual ~CommandEncoder() = default;
    virtual std::unique_ptr<RenderPassEncoder>
    beginRenderPass(RenderTarget* target) = 0;
    virtual std::unique_ptr<CommandBuffer> finish() = 0;
};

class Framebuffer {
  public:
    virtual ~Framebuffer() = default;
    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual void resize(int scrWidth, int scrHeight) = 0;
    virtual void resolve() = 0;
    virtual void blitToScreen(int scrWidth, int scrHeight) = 0;
    virtual std::vector<uint8_t> readColorPixels(bool flipY = true) = 0;
    virtual std::vector<uint8_t> readColorPixelsResized(int width, int height,
                                                        bool flipY = true) = 0;
    virtual Texture* getColorTexture() = 0;
    virtual Texture* getDepthTexture() = 0;
    virtual Texture* getStencilTexture() = 0;
    virtual Texture* getDepthStencilTexture() = 0;
};

class Shader {
  public:
    virtual ~Shader() = default;
    // Stable diagnostic identity and backend-neutral source descriptor.
    // Materials use the descriptor to request cached RHI pipeline variants;
    // render code must not infer shader behavior from file names.
    virtual const std::string& getName() const = 0;
    virtual const ShaderDesc& getDesc() const = 0;
    virtual void bind() = 0;
    virtual void unbind() = 0;

    // Legacy immediate-uniform API.
    //
    // This maps naturally to OpenGL uniform calls and remains useful for
    // debug shaders and the existing forward renderer. WebGPU/Vulkan-style
    // paths should treat it as compatibility glue, not as the long-term
    // material/pass binding model.

    // KE::Shader compatibility
    virtual void use() = 0; // Alias for bind()

    // Uniform setters - KE::Shader compatible
    virtual void setBool(const std::string& name, bool value) = 0;
    virtual void setInt(const std::string& name, int value) = 0;
    virtual void setFloat(const std::string& name, float value) = 0;
    virtual void setColor(const std::string& name, float r, float g, float b,
                          float a) = 0;

    virtual void setVec2(const std::string& name, const glm::vec2& value) = 0;
    virtual void setVec2(const std::string& name, float x, float y) = 0;
    virtual void setVec3(const std::string& name, const glm::vec3& value) = 0;
    virtual void setVec3(const std::string& name, float x, float y,
                         float z) = 0;
    virtual void setVec4(const std::string& name, const glm::vec4& value) = 0;
    virtual void setVec4(const std::string& name, float x, float y, float z,
                         float w) = 0;
    virtual void setMat2(const std::string& name, const glm::mat2& value) = 0;
    virtual void setMat3(const std::string& name, const glm::mat3& value) = 0;
    virtual void setMat4(const std::string& name, const glm::mat4& value) = 0;
    virtual void setMat4Array(const std::string& name, const glm::mat4* values,
                              size_t count) = 0;

    // Legacy OpenGL-style UBO binding by shader block name.
    virtual void setUniformBlockBinding(const std::string& blockName,
                                        int slot) = 0;
};

class Texture {
  public:
    virtual ~Texture() = default;
    virtual void bind(int slot = 0) = 0;
    virtual void unbind() = 0;
    virtual int getWidth() const = 0;
    virtual int getHeight() const = 0;
    virtual TextureFormat getFormat() const = 0;
    virtual TextureUsage getUsage() const = 0;
    virtual uint32_t getMipLevelCount() const = 0;
    virtual uint32_t getSampleCount() const = 0;
    virtual uint32_t getDepthOrArrayLayers() const = 0;
    virtual TextureDimension getDimension() const = 0;
    // Backend-native resource handle (GLuint, etc.) for external use
    virtual uintptr_t getNativeHandle() const = 0;
};

class TextureView {
  public:
    virtual ~TextureView() = default;
    virtual Texture* getTexture() const = 0;
    virtual const TextureViewDesc& getDesc() const = 0;
};

class Sampler {
  public:
    virtual ~Sampler() = default;
};

class RenderTarget {
  public:
    virtual ~RenderTarget() = default;
    virtual const RenderPassDesc& getDesc() const = 0;
    virtual int getWidth() const = 0;
    virtual int getHeight() const = 0;
    virtual uint32_t getSampleCount() const = 0;
};

class GraphicsPipeline {
  public:
    virtual ~GraphicsPipeline() = default;
    virtual const GraphicsPipelineDesc& getDesc() const = 0;
};

class BindGroupLayout {
  public:
    virtual ~BindGroupLayout() = default;
    virtual const BindGroupLayoutDesc& getDesc() const = 0;
};

class PipelineLayout {
  public:
    virtual ~PipelineLayout() = default;
    virtual const PipelineLayoutDesc& getDesc() const = 0;
};

class BindGroup {
  public:
    virtual ~BindGroup() = default;
    virtual const BindGroupDesc& getDesc() const = 0;
};

class VertexArray {
  public:
    virtual ~VertexArray() = default;
    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual void setVertexAttribute(const VertexAttribute& attribute) = 0;
    virtual void setVertexBuffer(Buffer* buffer) = 0;
    virtual void setIndexBuffer(Buffer* buffer) = 0;
};

} // namespace Backend
} // namespace KE

#endif
