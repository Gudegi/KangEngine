///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _OPENGL_DEVICE_HPP_
#define _OPENGL_DEVICE_HPP_

#include <cstddef>
#include <glad/glad.h>
#include <memory>
#include "../base/graphics_device.hpp"

namespace KE {
namespace Backend {

class OpenGLBuffer : public Buffer {
  private:
    GLuint _buffer;
    GLenum _target;
    BufferType _type;
    size_t _size;

  public:
    OpenGLBuffer(BufferType type, size_t size, const void* data = nullptr);
    ~OpenGLBuffer() override;

    void bind() override;
    void unbind() override;
    void setData(const void* data, size_t size, size_t offset = 0) override;
    BufferType getType() const override { return _type; }
};

class OpenGLShader : public Shader {
  private:
    GLuint _shaderProgram;
    std::string _name;

    std::string loadFile(const std::string& path);
    GLuint compile(const std::string& source, GLenum type);
    GLuint link(GLuint vertexShader, GLuint fragmentShader);
    void checkCompileError(GLuint shader);
    void checkLinkError(GLuint shaderProgram);

  public:
    OpenGLShader(const ShaderDesc& desc);
    ~OpenGLShader() override;

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
};

class OpenGLTexture : public Texture {
  private:
    GLuint _textureID;
    GLenum _target = GL_TEXTURE_2D;
    int _width, _height, _channels;
    float _warpParam, _filterMinParam, _filterMaxParam;

  public:
    OpenGLTexture(const TextureDesc& desc);
    OpenGLTexture(const TextureDesc& desc, float warpParam,
                  float filterMinParam, float filterMaxParam);
    ~OpenGLTexture() override;

    void bind(int slot = 0) override;
    void unbind() override;
    void setWarpParam(GLfloat warpParam = GL_REPEAT) const;
    void setFilterParam(GLfloat filterMinParam = GL_LINEAR_MIPMAP_LINEAR,
                        GLfloat filterMaxParam = GL_LINEAR) const;
    int getWidth() const override { return _width; }
    int getHeight() const override { return _height; }
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
    GLuint _colorTex = 0; // GL_RGB   color texture
    GLuint _depthTex = 0; // GL_DEPTH_COMPONENT32 depth texture

    // --- [SIMPLE RBO] non-MSAA depth+stencil renderbuffer ---
    // GLuint _rbo = 0; // GL_DEPTH24_STENCIL8
    // attach: glFramebufferRenderbuffer(GL_DEPTH_STENCIL_ATTACHMENT, _rbo)
    // faster than texture when depth sampling not needed

    // --- MSAA FBO (RBO-based) ---
    GLuint _msaaFbo = 0;
    GLuint _msaaColorRbo = 0; // GL_RGB8
    GLuint _msaaDepthRbo = 0; // GL_DEPTH_COMPONENT32

    FramebufferDesc _desc;

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
};

class OpenGLDevice : public GraphicsDevice {
  private:
    bool _initialized;

  public:
    OpenGLDevice();
    ~OpenGLDevice() override;

    void initialize() override;
    void shutdown() override;
    BackendType getBackendType() const override { return BackendType::OpenGL; }

    // Rendering
    void beginFrame() override;
    void endFrame() override;
    void clear(float r, float g, float b, float a) override;
    void setViewport(int x, int y, int width, int height) override;
    void drawIndexed(size_t indexCount) override;
    void drawIndexedInstanced(size_t indexCount, size_t instanceCount) override;
    void checkError() override;

    // Render State
    void setDepthTest(bool enable) override;
    void setDepthWrite(bool enable) override;
    void setBlend(bool enable) override;
    void setBlendFunc(BlendFactor src, BlendFactor dst) override;
    void setStencilTest(bool enable) override;
    void setPolygonMode(PolygonMode mode) override;
    void setClearColor(float r, float g, float b, float a) override;

    // Resource creation
    std::unique_ptr<Buffer> createBuffer(BufferType type, size_t size,
                                         const void* data = nullptr) override;
    std::unique_ptr<Shader> createShader(const ShaderDesc& desc) override;
    std::unique_ptr<Texture> createTexture(const TextureDesc& desc) override;
    std::unique_ptr<VertexArray> createVertexArray() override;

    // Convenience shader creation methods (KE::Shader compatible)
    std::unique_ptr<Shader> createShader(const char* vertexSource,
                                         const char* fragmentSource) override;
    std::unique_ptr<Shader>
    createShader(const std::string& vertexSource,
                 const std::string& fragmentSource) override;
    std::unique_ptr<Texture> createTexture(const std::string path,
                                           bool flip = false) override;
    std::unique_ptr<Texture>
    createTexture(const std::string path, bool flip = false,
                  float warpParam = GL_REPEAT,
                  float minFilferParam = GL_LINEAR_MIPMAP_LINEAR,
                  float maxFilterParam = GL_LINEAR) override;
    std::unique_ptr<Framebuffer>
    createFramebuffer(const FramebufferDesc& desc) override;
};

} // namespace Backend
} // namespace KE

#endif