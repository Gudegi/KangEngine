///
/// Author Kyungwon Kang, 2024/11
///

#include "opengl_device.hpp"
#include "../base/base_utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <glm/gtc/type_ptr.hpp>
#include <stb_image.h>

namespace KE {
namespace Backend {

// OpenGLBuffer Implementation
OpenGLBuffer::OpenGLBuffer(BufferType type, size_t size, const void* data)
    : _type(type), _size(size) {

    switch(type) {
        case BufferType::Vertex:
            _target = GL_ARRAY_BUFFER;
            break;
        case BufferType::Index:
            _target = GL_ELEMENT_ARRAY_BUFFER;
            break;
        case BufferType::Uniform:
            _target = GL_UNIFORM_BUFFER;
            break;
    }

    glGenBuffers(1, &_buffer);
    glBindBuffer(_target, _buffer);
    glBufferData(_target, size, data, GL_STATIC_DRAW);
    glBindBuffer(_target, 0);
}

OpenGLBuffer::~OpenGLBuffer() {
    glDeleteBuffers(1, &_buffer);
}

void OpenGLBuffer::bind() {
    glBindBuffer(_target, _buffer);
}

void OpenGLBuffer::unbind() {
    glBindBuffer(_target, 0);
}

void OpenGLBuffer::setData(const void* data, size_t size, size_t offset) {
    glBindBuffer(_target, _buffer);
    glBufferSubData(_target, offset, size, data);
    glBindBuffer(_target, 0);
}

// OpenGLShader Implementation
OpenGLShader::OpenGLShader(const ShaderDesc& desc) : _name(desc.name) {
    GLuint vertexShader = 0, fragmentShader = 0;

    for (const auto& stage : desc.stages) {
        if (stage.type == ShaderType::Vertex) {
            vertexShader = compile(stage.source, GL_VERTEX_SHADER);
        } else if (stage.type == ShaderType::Fragment) {
            fragmentShader = compile(stage.source, GL_FRAGMENT_SHADER);
        }
    }

    if (vertexShader == 0 || fragmentShader == 0) {
        std::cerr << "Error: Missing vertex or fragment shader" << std::endl;
        return;
    }

    _shaderProgram = link(vertexShader, fragmentShader);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

OpenGLShader::~OpenGLShader() {
    glDeleteProgram(_shaderProgram);
}

std::string OpenGLShader::loadFile(const std::string& path) {
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint OpenGLShader::compile(const std::string& source, GLenum type) {
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    checkCompileError(shader);
    return shader;
}

GLuint OpenGLShader::link(GLuint vertexShader, GLuint fragmentShader) {
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    checkLinkError(shaderProgram);
    return shaderProgram;
}

void OpenGLShader::checkCompileError(GLuint shader) {
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
    }
}

void OpenGLShader::checkLinkError(GLuint shaderProgram) {
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader linking failed: " << infoLog << std::endl;
    }
}

void OpenGLShader::bind() {
    glUseProgram(_shaderProgram);
}

void OpenGLShader::unbind() {
    glUseProgram(0);
}

void OpenGLShader::setInt(const std::string& name, int value) {
    glUniform1i(glGetUniformLocation(_shaderProgram, name.c_str()), value);
}

void OpenGLShader::setFloat(const std::string& name, float value) {
    glUniform1f(glGetUniformLocation(_shaderProgram, name.c_str()), value);
}

void OpenGLShader::setVec2(const std::string& name, const glm::vec2& value) {
    glUniform2fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, &value[0]);
}

void OpenGLShader::setVec3(const std::string& name, const glm::vec3& value) {
    glUniform3fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, &value[0]);
}

void OpenGLShader::setVec4(const std::string& name, const glm::vec4& value) {
    glUniform4fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, &value[0]);
}

void OpenGLShader::setMat3(const std::string& name, const glm::mat3& value) {
    glUniformMatrix3fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, GL_FALSE, &value[0][0]);
}

void OpenGLShader::setMat4(const std::string& name, const glm::mat4& value) {
    glUniformMatrix4fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, GL_FALSE, &value[0][0]);
}

// KE::Shader compatibility methods
void OpenGLShader::use() {
    bind();  // Alias for bind()
}

void OpenGLShader::setBool(const std::string& name, bool value) {
    glUniform1i(glGetUniformLocation(_shaderProgram, name.c_str()), static_cast<int>(value));
}

void OpenGLShader::setColor(const std::string& name, float r, float g, float b, float a) {
    glUniform4f(glGetUniformLocation(_shaderProgram, name.c_str()), r, g, b, a);
}

void OpenGLShader::setVec2(const std::string& name, float x, float y) {
    glUniform2f(glGetUniformLocation(_shaderProgram, name.c_str()), x, y);
}

void OpenGLShader::setVec3(const std::string& name, float x, float y, float z) {
    glUniform3f(glGetUniformLocation(_shaderProgram, name.c_str()), x, y, z);
}

void OpenGLShader::setVec4(const std::string& name, float x, float y, float z, float w) {
    glUniform4f(glGetUniformLocation(_shaderProgram, name.c_str()), x, y, z, w);
}

void OpenGLShader::setMat2(const std::string& name, const glm::mat2& value) {
    glUniformMatrix2fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, GL_FALSE, &value[0][0]);
}

// OpenGLTexture Implementation
OpenGLTexture::OpenGLTexture(const TextureDesc& desc)
    : _width(desc.width), _height(desc.height), _channels(desc.channels) {

    glGenTextures(1, &_textureID);
    glBindTexture(_target, _textureID);

    GLenum format;
    switch (_channels) {
    case 4: format = GL_RGBA; break;
    case 3: format = GL_RGB; break;
    case 1: format = GL_RED; break;
    default:
        std::cerr << "Unsupported channel count: " << _channels << std::endl;
        format = GL_RGBA; 
    }

    glTexImage2D(_target, 0, format, _width, _height, 0, format, GL_UNSIGNED_BYTE, desc.data);
    glGenerateMipmap(_target);

    setWarpParam();
    setFilterParam(); 

    glBindTexture(_target, 0);
}

OpenGLTexture::OpenGLTexture(const TextureDesc& desc, float warpParam, float filterMinParam, float filterMaxParam)
    : _width(desc.width), _height(desc.height), _channels(desc.channels), 
        _warpParam(warpParam), _filterMinParam(filterMinParam), _filterMaxParam(filterMaxParam) {

    glGenTextures(1, &_textureID);
    glBindTexture(_target, _textureID);

    GLenum format;
    switch (_channels) {
    case 4: format = GL_RGBA; break;
    case 3: format = GL_RGB; break;
    case 1: format = GL_RED; break;
    default:
        std::cerr << "Unsupported channel count: " << _channels << std::endl;
        format = GL_RGBA; 
    }

    glTexImage2D(_target, 0, format, _width, _height, 0, format, GL_UNSIGNED_BYTE, desc.data);
    glGenerateMipmap(_target);

    setWarpParam((GLfloat)_warpParam);
    setFilterParam((GLfloat)_filterMinParam, (GLfloat)_filterMaxParam); 

    glBindTexture(_target, 0);
}

OpenGLTexture::~OpenGLTexture() {
    glDeleteTextures(1, &_textureID);
}

void OpenGLTexture::bind(int slot) {
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(_target, _textureID);
}

void OpenGLTexture::unbind() {
    glBindTexture(_target, 0);
}

void OpenGLTexture::setWarpParam(GLfloat warpParam) const
{
    glTexParameteri(_target, GL_TEXTURE_WRAP_S, warpParam);
    glTexParameteri(_target, GL_TEXTURE_WRAP_T, warpParam);
}

void OpenGLTexture::setFilterParam(GLfloat filterMinParam, GLfloat filterMaxParam) const
{
    glTexParameteri(_target, GL_TEXTURE_MIN_FILTER, filterMinParam);
    glTexParameteri(_target, GL_TEXTURE_MAG_FILTER, filterMaxParam);
}

// OpenGLDevice Implementation
OpenGLDevice::OpenGLDevice() : _initialized(false) {
}

OpenGLDevice::~OpenGLDevice() {
    if (_initialized) {
        shutdown();
    }
}

void OpenGLDevice::initialize() {
    if (_initialized) return;

    // OpenGL context should already be created by Window
    glEnable(GL_DEPTH_TEST);
    _initialized = true;

    std::cout << "OpenGL Device initialized" << std::endl;
}

void OpenGLDevice::shutdown() {
    if (!_initialized) return;

    _initialized = false;
    std::cout << "OpenGL Device shutdown" << std::endl;
}

void OpenGLDevice::beginFrame() {
    // Nothing specific needed for OpenGL
}

void OpenGLDevice::endFrame() {
    // Nothing specific needed for OpenGL
}

void OpenGLDevice::clear(float r, float g, float b, float a) {
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLDevice::setViewport(int x, int y, int width, int height) {
    glViewport(x, y, width, height);
}

void OpenGLDevice::drawIndexed(int indexCount) {
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
}

void OpenGLDevice::checkError() {
    GLenum err;
    if ((err = glGetError()) != GL_NO_ERROR)
    {
        std::string errStr = "";
        switch (err)
        {
            case GL_INVALID_ENUM:      errStr = "GL_INVALID_ENUM"; break;
            case GL_INVALID_VALUE:     errStr = "GL_INVALID_VALUE"; break;
            case GL_INVALID_OPERATION: errStr = "GL_INVALID_OPERATION"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: errStr = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
            case GL_OUT_OF_MEMORY:   errStr = "GL_OUT_OF_MEMORY"; break;
            case GL_STACK_UNDERFLOW: errStr = "GL_STACK_UNDERFLOW"; break;
            case GL_STACK_OVERFLOW:  errStr = "GL_STACK_OVERFLOW"; break;
        }
        std::cout << errStr << std::endl;
    }
}

std::unique_ptr<Buffer> OpenGLDevice::createBuffer(BufferType type, size_t size, const void* data) {
    return std::make_unique<OpenGLBuffer>(type, size, data);
}

std::unique_ptr<Shader> OpenGLDevice::createShader(const ShaderDesc& desc) {
    return std::make_unique<OpenGLShader>(desc);
}

std::unique_ptr<Texture> OpenGLDevice::createTexture(const TextureDesc& desc) {
    return std::make_unique<OpenGLTexture>(desc);
}

std::unique_ptr<VertexArray> OpenGLDevice::createVertexArray() {
    return std::make_unique<OpenGLVertexArray>();
}

std::unique_ptr<Shader> OpenGLDevice::createShader(const char* vertexSource, const char* fragmentSource) {
    ShaderDesc desc;
    desc.name = "ConvenienceShader";
    desc.stages = {
        {std::string(vertexSource), ShaderType::Vertex},
        {std::string(fragmentSource), ShaderType::Fragment}
    };
    return std::make_unique<OpenGLShader>(desc);
}

std::unique_ptr<Shader> OpenGLDevice::createShader(const std::string& vertexSource, const std::string& fragmentSource) {
    return createShader(vertexSource.c_str(), fragmentSource.c_str());
}

std::unique_ptr<Texture> OpenGLDevice::createTexture(const std::string path, bool flip) {
    TextureDesc desc = loadImage(path, flip);
    auto texture = std::make_unique<OpenGLTexture>(desc);
    stbi_image_free((void*)desc.data); // release memory
    return texture;
}

std::unique_ptr<Texture> OpenGLDevice::createTexture(const std::string path, bool flip, float warpParam, float minFilferParam, float maxFilterParam) {
    TextureDesc desc = loadImage(path, flip);
    auto texture = std::make_unique<OpenGLTexture>(desc, warpParam, minFilferParam, maxFilterParam);
    stbi_image_free((void*)desc.data); // release memory
    return texture;
}

void OpenGLDevice::setDepthTest(bool enable) {
    if (enable) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
}

void OpenGLDevice::setPolygonMode(PolygonMode mode) {
    GLenum glMode;
    switch (mode) {
        case PolygonMode::Fill:
            glMode = GL_FILL;
            break;
        case PolygonMode::Line:
            glMode = GL_LINE;
            break;
        case PolygonMode::Point:
            glMode = GL_POINT;
            break;
    }
    glPolygonMode(GL_FRONT_AND_BACK, glMode);
}

void OpenGLDevice::setClearColor(float r, float g, float b, float a) {
    glClearColor(r, g, b, a);
}

// OpenGLVertexArray Implementation
OpenGLVertexArray::OpenGLVertexArray() {
    glGenVertexArrays(1, &_vao);
}

OpenGLVertexArray::~OpenGLVertexArray() {
    glDeleteVertexArrays(1, &_vao);
}

void OpenGLVertexArray::bind() {
    glBindVertexArray(_vao);
}

void OpenGLVertexArray::unbind() {
    glBindVertexArray(0);
}

void OpenGLVertexArray::setVertexAttribute(const VertexAttribute& attribute) {
    bind();

    GLenum glType;
    switch (attribute.type) {
        case VertexAttributeType::Float:
            glType = GL_FLOAT;
            break;
        case VertexAttributeType::Int:
            glType = GL_INT;
            break;
        case VertexAttributeType::UnsignedInt:
            glType = GL_UNSIGNED_INT;
            break;
        case VertexAttributeType::Byte:
            glType = GL_BYTE;
            break;
        case VertexAttributeType::UnsignedByte:
            glType = GL_UNSIGNED_BYTE;
            break;
    }

    glVertexAttribPointer(
        attribute.location,
        attribute.size,
        glType,
        attribute.normalized ? GL_TRUE : GL_FALSE,
        attribute.stride,
        reinterpret_cast<void*>(attribute.offset)
    );
    glEnableVertexAttribArray(attribute.location);
}

void OpenGLVertexArray::setVertexBuffer(Buffer* buffer) {
    bind();
    buffer->bind();
}

void OpenGLVertexArray::setIndexBuffer(Buffer* buffer) {
    bind();
    buffer->bind();
}

} // namespace Backend
} // namespace KE