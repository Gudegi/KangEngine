///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _OPENGL_DEVICE_HPP_
#define _OPENGL_DEVICE_HPP_

#include <glad/glad.h>
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
    void setColor(const std::string& name, float r, float g, float b, float a) override;

    void setVec2(const std::string& name, const glm::vec2& value) override;
    void setVec2(const std::string& name, float x, float y) override;
    void setVec3(const std::string& name, const glm::vec3& value) override;
    void setVec3(const std::string& name, float x, float y, float z) override;
    void setVec4(const std::string& name, const glm::vec4& value) override;
    void setVec4(const std::string& name, float x, float y, float z, float w) override;
    void setMat2(const std::string& name, const glm::mat2& value) override;
    void setMat3(const std::string& name, const glm::mat3& value) override;
    void setMat4(const std::string& name, const glm::mat4& value) override;
};

class OpenGLTexture : public Texture {
private:
    GLuint _texture;
    int _width, _height;
    int _channels;

public:
    OpenGLTexture(const TextureDesc& desc);
    ~OpenGLTexture() override;

    void bind(int slot = 0) override;
    void unbind() override;
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
    void drawIndexed(int indexCount) override;

    // Render State
    void setDepthTest(bool enable) override;
    void setPolygonMode(PolygonMode mode) override;
    void setClearColor(float r, float g, float b, float a) override;

    // Resource creation
    std::unique_ptr<Buffer> createBuffer(BufferType type, size_t size, const void* data = nullptr) override;
    std::unique_ptr<Shader> createShader(const ShaderDesc& desc) override;
    std::unique_ptr<Texture> createTexture(const TextureDesc& desc) override;
    std::unique_ptr<VertexArray> createVertexArray() override;

    // Convenience shader creation methods (KE::Shader compatible)
    std::unique_ptr<Shader> createShader(const char* vertexSource, const char* fragmentSource) override;
    std::unique_ptr<Shader> createShader(const std::string& vertexSource, const std::string& fragmentSource) override;
};

} // namespace Backend
} // namespace KE

#endif