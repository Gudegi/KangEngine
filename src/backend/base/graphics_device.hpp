///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _GRAPHICS_DEVICE_HPP_
#define _GRAPHICS_DEVICE_HPP_

#include <glad/glad.h>
#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>

namespace KE {
namespace Backend {

using std::string;
using std::unique_ptr;
using std::vector;

enum class BackendType { OpenGL, Vulkan, WebGPU };

enum class BufferType { Vertex, DynamicVertex, Index, Uniform };

enum class ShaderType { Vertex, Fragment, Geometry, Compute };

enum class UniformType { Int, Float, Vec2, Vec3, Vec4, Mat3, Mat4 };

enum class VertexAttributeType { Float, Int, UnsignedInt, Byte, UnsignedByte };

enum class PolygonMode { Fill, Line, Point };

enum class BlendFactor {
    Zero,
    One,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
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

struct TextureDesc {
    int width, height;
    int channels = 4;
    const void* data = nullptr;
    std::string name;
};

struct FramebufferDesc {
    int width, height;
    bool depthOnly = false; // shadow FBO: no color attachment
    bool stencil = false;   // use depth+stencil
    int msaaSamples = 0;    // 0 means No MSAA
};

// Forward declarations
class Buffer;
class Shader;
class Texture;
class VertexArray;
class Framebuffer;

class GraphicsDevice {
  public:
    virtual ~GraphicsDevice() = default;

    virtual void initialize() = 0;
    virtual void shutdown() = 0;
    virtual BackendType getBackendType() const = 0;

    // Rendering
    virtual void beginFrame() = 0;
    virtual void endFrame() = 0;
    virtual void clear(float r, float g, float b, float a) = 0;
    virtual void setViewport(int x, int y, int width, int height) = 0;
    virtual void drawIndexed(size_t indexCount) = 0;
    // instanced rendering
    virtual void drawIndexedInstanced(size_t indexCount,
                                      size_t instanceCount) = 0;
    virtual void checkError() = 0;

    // Render State
    virtual void setDepthTest(bool enable) = 0;
    virtual void setDepthWrite(bool enable) = 0;
    virtual void setBlend(bool enable) = 0;
    virtual void setBlendFunc(BlendFactor src, BlendFactor dst) = 0;
    virtual void setStencilTest(bool enable) = 0;
    virtual void setPolygonMode(PolygonMode mode) = 0;
    virtual void setClearColor(float r, float g, float b, float a) = 0;

    // Resource creation
    virtual std::unique_ptr<Buffer>
    createBuffer(BufferType type, size_t size, const void* data = nullptr) = 0;
    virtual void bindUniformBuffer(Buffer* buffer, int slot) = 0;
    virtual std::unique_ptr<Shader> createShader(const ShaderDesc& desc) = 0;
    virtual std::unique_ptr<Texture> createTexture(const TextureDesc& desc) = 0;
    virtual std::unique_ptr<VertexArray> createVertexArray() = 0;

    // OpenGL specific functions
    virtual std::unique_ptr<Shader>
    createShader(const char* vertexSource, const char* fragmentSource) = 0;
    virtual std::unique_ptr<Shader>
    createShader(const std::string& vertexSource,
                 const std::string& fragmentSource) = 0;
    virtual std::unique_ptr<Texture> createTexture(const std::string path,
                                                   bool flip = false) = 0;
    virtual std::unique_ptr<Texture> createTexture(const std::string path,
                                                   bool flip, float warpParam,
                                                   float minFilferParam,
                                                   float maxFilterParam) = 0;

    virtual std::unique_ptr<Framebuffer>
    createFramebuffer(const FramebufferDesc& desc) = 0;
};

class Buffer {
  public:
    virtual ~Buffer() = default;
    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual void setData(const void* data, size_t size, size_t offset = 0) = 0;
    virtual BufferType getType() const = 0;
};

class Framebuffer {
  public:
    virtual ~Framebuffer() = default;
    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual void resize(int scrWidth, int scrHeight) = 0;
    virtual void resolve() = 0;
    virtual void blitToScreen(int scrWidth, int scrHeight) = 0;
    virtual Texture* getColorTexture() = 0;
    virtual Texture* getDepthTexture() = 0;
    virtual Texture* getStencilTexture() = 0;
    virtual Texture* getDepthStencilTexture() = 0;
};

class Shader {
  public:
    virtual ~Shader() = default;
    virtual void bind() = 0;
    virtual void unbind() = 0;

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

    // for OpenGL UBO binding
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