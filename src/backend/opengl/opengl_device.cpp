///
/// Author Kyungwon Kang, 2024/11
///

#include "opengl_device.hpp"
#include "../base/base_utils.hpp"
#include <cstddef>
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

    switch (type) {
    case BufferType::Vertex:
    case BufferType::DynamicVertex:
        _target = GL_ARRAY_BUFFER;
        break;
    case BufferType::Index:
        _target = GL_ELEMENT_ARRAY_BUFFER;
        break;
    case BufferType::Uniform:
        _target = GL_UNIFORM_BUFFER;
        break;
    }

    GLenum usage = (type == BufferType::Uniform ||
                    type == BufferType::DynamicVertex)
                       ? GL_DYNAMIC_DRAW
                       : GL_STATIC_DRAW;
    glGenBuffers(1, &_buffer);
    glBindBuffer(_target, _buffer);
    glBufferData(_target, size, data, usage);
    glBindBuffer(_target, 0);
}

OpenGLBuffer::~OpenGLBuffer() { glDeleteBuffers(1, &_buffer); }

void OpenGLBuffer::bind() { glBindBuffer(_target, _buffer); }

void OpenGLBuffer::unbind() { glBindBuffer(_target, 0); }

void OpenGLBuffer::setData(const void* data, size_t size, size_t offset) {
    if (offset + size > _size) {
        std::cerr << "OpenGLBuffer::setData: out of bounds (offset=" << offset
                  << " size=" << size << " buffer_size=" << _size << ")\n";
        return;
    }
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

OpenGLShader::~OpenGLShader() { glDeleteProgram(_shaderProgram); }

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

void OpenGLShader::bind() { glUseProgram(_shaderProgram); }

void OpenGLShader::unbind() { glUseProgram(0); }

void OpenGLShader::setInt(const std::string& name, int value) {
    glUniform1i(glGetUniformLocation(_shaderProgram, name.c_str()), value);
}

void OpenGLShader::setFloat(const std::string& name, float value) {
    glUniform1f(glGetUniformLocation(_shaderProgram, name.c_str()), value);
}

void OpenGLShader::setVec2(const std::string& name, const glm::vec2& value) {
    glUniform2fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1,
                 &value[0]);
}

void OpenGLShader::setVec3(const std::string& name, const glm::vec3& value) {
    glUniform3fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1,
                 &value[0]);
}

void OpenGLShader::setVec4(const std::string& name, const glm::vec4& value) {
    glUniform4fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1,
                 &value[0]);
}

void OpenGLShader::setMat3(const std::string& name, const glm::mat3& value) {
    glUniformMatrix3fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1,
                       GL_FALSE, &value[0][0]);
}

void OpenGLShader::setMat4(const std::string& name, const glm::mat4& value) {
    glUniformMatrix4fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1,
                       GL_FALSE, &value[0][0]);
}

// KE::Shader compatibility methods
void OpenGLShader::use() {
    bind(); // Alias for bind()
}

void OpenGLShader::setBool(const std::string& name, bool value) {
    glUniform1i(glGetUniformLocation(_shaderProgram, name.c_str()),
                static_cast<int>(value));
}

void OpenGLShader::setColor(const std::string& name, float r, float g, float b,
                            float a) {
    glUniform4f(glGetUniformLocation(_shaderProgram, name.c_str()), r, g, b, a);
}

void OpenGLShader::setVec2(const std::string& name, float x, float y) {
    glUniform2f(glGetUniformLocation(_shaderProgram, name.c_str()), x, y);
}

void OpenGLShader::setVec3(const std::string& name, float x, float y, float z) {
    glUniform3f(glGetUniformLocation(_shaderProgram, name.c_str()), x, y, z);
}

void OpenGLShader::setVec4(const std::string& name, float x, float y, float z,
                           float w) {
    glUniform4f(glGetUniformLocation(_shaderProgram, name.c_str()), x, y, z, w);
}

void OpenGLShader::setMat2(const std::string& name, const glm::mat2& value) {
    glUniformMatrix2fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1,
                       GL_FALSE, &value[0][0]);
}

void OpenGLShader::setUniformBlockBinding(const std::string& blockName,
                                          int slot) {
    GLuint uniformBlockIndex =
        glGetUniformBlockIndex(_shaderProgram, blockName.c_str());
    if (uniformBlockIndex == GL_INVALID_INDEX) {
        std::cerr << "OpenGLShader::setUniformBlockBinding: block '"
                  << blockName << "' not found in shader\n";
        return;
    }
    glUniformBlockBinding(_shaderProgram, uniformBlockIndex, slot);
}

// OpenGLTexture Implementation
OpenGLTexture::OpenGLTexture(const TextureDesc& desc)
    : _width(desc.width), _height(desc.height), _channels(desc.channels) {

    glGenTextures(1, &_textureID);
    glBindTexture(_target, _textureID);

    GLenum format;
    switch (_channels) {
    case 4:
        format = GL_RGBA;
        break;
    case 3:
        format = GL_RGB;
        break;
    case 1:
        format = GL_RED;
        break;
    default:
        std::cerr << "Unsupported channel count: " << _channels << std::endl;
        format = GL_RGBA;
    }

    glTexImage2D(_target, 0, format, _width, _height, 0, format,
                 GL_UNSIGNED_BYTE, desc.data);
    glGenerateMipmap(_target);

    setWarpParam();
    setFilterParam();

    glBindTexture(_target, 0);
}

OpenGLTexture::OpenGLTexture(const TextureDesc& desc, float warpParam,
                             float filterMinParam, float filterMaxParam)
    : _width(desc.width), _height(desc.height), _channels(desc.channels),
      _warpParam(warpParam), _filterMinParam(filterMinParam),
      _filterMaxParam(filterMaxParam) {

    glGenTextures(1, &_textureID);
    glBindTexture(_target, _textureID);

    GLenum format;
    switch (_channels) {
    case 4:
        format = GL_RGBA;
        break;
    case 3:
        format = GL_RGB;
        break;
    case 1:
        format = GL_RED;
        break;
    default:
        std::cerr << "Unsupported channel count: " << _channels << std::endl;
        format = GL_RGBA;
    }

    glTexImage2D(_target, 0, format, _width, _height, 0, format,
                 GL_UNSIGNED_BYTE, desc.data);
    glGenerateMipmap(_target);

    setWarpParam((GLfloat)_warpParam);
    setFilterParam((GLfloat)_filterMinParam, (GLfloat)_filterMaxParam);

    glBindTexture(_target, 0);
}

OpenGLTexture::~OpenGLTexture() { glDeleteTextures(1, &_textureID); }

void OpenGLTexture::bind(int slot) {
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(_target, _textureID);
}

void OpenGLTexture::unbind() { glBindTexture(_target, 0); }

void OpenGLTexture::setWarpParam(GLfloat warpParam) const {
    glTexParameteri(_target, GL_TEXTURE_WRAP_S, warpParam);
    glTexParameteri(_target, GL_TEXTURE_WRAP_T, warpParam);
}

void OpenGLTexture::setFilterParam(GLfloat filterMinParam,
                                   GLfloat filterMaxParam) const {
    glTexParameteri(_target, GL_TEXTURE_MIN_FILTER, filterMinParam);
    glTexParameteri(_target, GL_TEXTURE_MAG_FILTER, filterMaxParam);
}

// OpenGLDevice Implementation
OpenGLDevice::OpenGLDevice() : _initialized(false) {}

OpenGLDevice::~OpenGLDevice() {
    if (_initialized) {
        shutdown();
    }
}

void OpenGLDevice::initialize() {
    if (_initialized)
        return;

    // OpenGL context should already be created by Window
    glEnable(GL_DEPTH_TEST);
    // TODO: need material-wise different culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    _initialized = true;

    std::cout << "OpenGL Device initialized" << std::endl;
}

void OpenGLDevice::shutdown() {
    if (!_initialized)
        return;

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
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void OpenGLDevice::setViewport(int x, int y, int width, int height) {
    glViewport(x, y, width, height);
}

void OpenGLDevice::drawIndexed(size_t indexCount) {
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indexCount),
                   GL_UNSIGNED_INT, 0);
}

void OpenGLDevice::drawIndexedInstanced(size_t indexCount,
                                        size_t instanceCount) {
    glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(indexCount),
                            GL_UNSIGNED_INT, 0,
                            static_cast<GLsizei>(instanceCount));
}

void OpenGLDevice::checkError() {
    GLenum err;
    if ((err = glGetError()) != GL_NO_ERROR) {
        std::string errStr = "";
        switch (err) {
        case GL_INVALID_ENUM:
            errStr = "GL_INVALID_ENUM";
            break;
        case GL_INVALID_VALUE:
            errStr = "GL_INVALID_VALUE";
            break;
        case GL_INVALID_OPERATION:
            errStr = "GL_INVALID_OPERATION";
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            errStr = "GL_INVALID_FRAMEBUFFER_OPERATION";
            break;
        case GL_OUT_OF_MEMORY:
            errStr = "GL_OUT_OF_MEMORY";
            break;
        case GL_STACK_UNDERFLOW:
            errStr = "GL_STACK_UNDERFLOW";
            break;
        case GL_STACK_OVERFLOW:
            errStr = "GL_STACK_OVERFLOW";
            break;
        }
        std::cout << errStr << std::endl;
    }
}

std::unique_ptr<Buffer> OpenGLDevice::createBuffer(BufferType type, size_t size,
                                                   const void* data) {
    return std::make_unique<OpenGLBuffer>(type, size, data);
}

void OpenGLDevice::bindUniformBuffer(Buffer* buffer, int slot) {
    auto* glBuf = static_cast<OpenGLBuffer*>(buffer);
    glBindBufferBase(GL_UNIFORM_BUFFER, slot, glBuf->getHandle());
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

std::unique_ptr<Shader> OpenGLDevice::createShader(const char* vertexSource,
                                                   const char* fragmentSource) {
    ShaderDesc desc;
    desc.name = "ConvenienceShader";
    desc.stages = {{std::string(vertexSource), ShaderType::Vertex},
                   {std::string(fragmentSource), ShaderType::Fragment}};
    return std::make_unique<OpenGLShader>(desc);
}

std::unique_ptr<Shader>
OpenGLDevice::createShader(const std::string& vertexSource,
                           const std::string& fragmentSource) {
    return createShader(vertexSource.c_str(), fragmentSource.c_str());
}

std::unique_ptr<Texture> OpenGLDevice::createTexture(const std::string path,
                                                     bool flip) {
    TextureDesc desc = loadImage(path, flip);
    auto texture = std::make_unique<OpenGLTexture>(desc);
    stbi_image_free((void*)desc.data); // release memory
    return texture;
}

std::unique_ptr<Texture> OpenGLDevice::createTexture(const std::string path,
                                                     bool flip, float warpParam,
                                                     float minFilferParam,
                                                     float maxFilterParam) {
    TextureDesc desc = loadImage(path, flip);
    auto texture = std::make_unique<OpenGLTexture>(
        desc, warpParam, minFilferParam, maxFilterParam);
    stbi_image_free((void*)desc.data); // release memory
    return texture;
}

void OpenGLDevice::setDepthTest(bool enable) {
    enable ? glEnable(GL_DEPTH_TEST) : glDisable(GL_DEPTH_TEST);
}

void OpenGLDevice::setDepthWrite(bool enable) { glDepthMask(enable); }

void OpenGLDevice::setBlend(bool enable) {
    enable ? glEnable(GL_BLEND) : glDisable(GL_BLEND);
}

void OpenGLDevice::setBlendFunc(BlendFactor src, BlendFactor dst) {
    auto toGL = [](BlendFactor f) -> GLenum {
        switch (f) {
        case BlendFactor::Zero:
            return GL_ZERO;
        case BlendFactor::One:
            return GL_ONE;
        case BlendFactor::SrcAlpha:
            return GL_SRC_ALPHA;
        case BlendFactor::OneMinusSrcAlpha:
            return GL_ONE_MINUS_SRC_ALPHA;
        case BlendFactor::DstAlpha:
            return GL_DST_ALPHA;
        case BlendFactor::OneMinusDstAlpha:
            return GL_ONE_MINUS_DST_ALPHA;
        }
        return GL_ONE;
    };
    glBlendFunc(toGL(src), toGL(dst));
}

void OpenGLDevice::setStencilTest(bool enable) {
    enable ? glEnable(GL_STENCIL_TEST) : glDisable(GL_STENCIL_TEST);
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

std::unique_ptr<Framebuffer>
OpenGLDevice::createFramebuffer(const FramebufferDesc& desc) {
    return std::make_unique<OpenGLFramebuffer>(desc);
}

// OpenGLVertexArray Implementation
OpenGLVertexArray::OpenGLVertexArray() { glGenVertexArrays(1, &_vao); }

OpenGLVertexArray::~OpenGLVertexArray() { glDeleteVertexArrays(1, &_vao); }

void OpenGLVertexArray::bind() { glBindVertexArray(_vao); }

void OpenGLVertexArray::unbind() { glBindVertexArray(0); }

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

    glVertexAttribPointer(attribute.location, attribute.size, glType,
                          attribute.normalized ? GL_TRUE : GL_FALSE,
                          attribute.stride,
                          reinterpret_cast<void*>(attribute.offset));
    glEnableVertexAttribArray(attribute.location);
    if (attribute.divisor > 0)
        glVertexAttribDivisor(attribute.location, attribute.divisor);
}

void OpenGLVertexArray::setVertexBuffer(Buffer* buffer) {
    bind();
    buffer->bind();
}

void OpenGLVertexArray::setIndexBuffer(Buffer* buffer) {
    bind();
    buffer->bind();
}

// OpenGLFramebuffer Impl
OpenGLFramebuffer::OpenGLFramebuffer(const FramebufferDesc& desc)
    : _desc(desc) {
    // ----------------------------------------------------------------
    // Texture FBO — always created; MSAA case uses this as resolve target
    // ----------------------------------------------------------------
    glGenFramebuffers(1, &_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);

    // color texture (GL_RGB)
    glGenTextures(1, &_colorTex);
    glBindTexture(GL_TEXTURE_2D, _colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _desc.width, _desc.height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           _colorTex, 0);

    // depth (or depth+stencil) texture — required for shadow map sampling
    // stencil=true: GL_DEPTH24_STENCIL8 + GL_DEPTH_STENCIL_ATTACHMENT
    // stencil=false: GL_DEPTH_COMPONENT32 + GL_DEPTH_ATTACHMENT
    GLenum depthFmt =
        _desc.stencil ? GL_DEPTH24_STENCIL8 : GL_DEPTH_COMPONENT32;
    GLenum depthAtt =
        _desc.stencil ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT;
    GLenum depthBase = _desc.stencil ? GL_DEPTH_STENCIL : GL_DEPTH_COMPONENT;
    GLenum depthType = _desc.stencil ? GL_UNSIGNED_INT_24_8 : GL_FLOAT;

    glGenTextures(1, &_depthTex);
    glBindTexture(GL_TEXTURE_2D, _depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, depthFmt, _desc.width, _desc.height, 0,
                 depthBase, depthType, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, depthAtt, GL_TEXTURE_2D, _depthTex,
                           0);

    // [SIMPLE RBO] depth+stencil renderbuffer (faster, but no shader sampling)
    // GLuint rbo;
    // glGenRenderbuffers(1, &rbo);
    // glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    // glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8,
    //                       _desc.width, _desc.height);
    // glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
    //                           GL_RENDERBUFFER, rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Texture FBO incomplete!"
                  << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ----------------------------------------------------------------
    // MSAA FBO
    // ----------------------------------------------------------------
    if (_desc.msaaSamples > 0) {
        // color RBO
        glGenRenderbuffers(1, &_msaaColorRbo);
        glBindRenderbuffer(GL_RENDERBUFFER, _msaaColorRbo);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, _desc.msaaSamples,
                                         GL_RGB8, _desc.width, _desc.height);

        // depth (or depth+stencil) RBO — format mirrors texture FBO
        GLenum msaaDepthFmt =
            _desc.stencil ? GL_DEPTH24_STENCIL8 : GL_DEPTH_COMPONENT32;
        GLenum msaaDepthAtt =
            _desc.stencil ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT;

        glGenRenderbuffers(1, &_msaaDepthRbo);
        glBindRenderbuffer(GL_RENDERBUFFER, _msaaDepthRbo);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, _desc.msaaSamples,
                                         msaaDepthFmt, _desc.width,
                                         _desc.height);

        // MSAA FBO
        glGenFramebuffers(1, &_msaaFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, _msaaFbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_RENDERBUFFER, _msaaColorRbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, msaaDepthAtt, GL_RENDERBUFFER,
                                  _msaaDepthRbo);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: MSAA FBO incomplete!"
                      << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

OpenGLFramebuffer::~OpenGLFramebuffer() {
    if (_msaaFbo)
        glDeleteFramebuffers(1, &_msaaFbo);
    if (_msaaColorRbo)
        glDeleteRenderbuffers(1, &_msaaColorRbo);
    if (_msaaDepthRbo)
        glDeleteRenderbuffers(1, &_msaaDepthRbo);
    if (_colorTex)
        glDeleteTextures(1, &_colorTex);
    if (_depthTex)
        glDeleteTextures(1, &_depthTex);
    if (_fbo)
        glDeleteFramebuffers(1, &_fbo);
}

void OpenGLFramebuffer::bind() {
    // Render into MSAA FBO if available, otherwise directly into texture FBO
    glBindFramebuffer(GL_FRAMEBUFFER, _desc.msaaSamples > 0 ? _msaaFbo : _fbo);
}

void OpenGLFramebuffer::unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

void OpenGLFramebuffer::resolve() {
    if (_desc.msaaSamples == 0)
        return;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, _msaaFbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _fbo);
    glBlitFramebuffer(0, 0, _desc.width, _desc.height, 0, 0, _desc.width,
                      _desc.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void OpenGLFramebuffer::resize(int w, int h) {
    _desc.width = w;
    _desc.height = h;

    // Resize texture FBO attachments
    glBindTexture(GL_TEXTURE_2D, _colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 nullptr);
    glBindTexture(GL_TEXTURE_2D, _depthTex);
    {
        GLenum depthFmt =
            _desc.stencil ? GL_DEPTH24_STENCIL8 : GL_DEPTH_COMPONENT32;
        GLenum depthBase =
            _desc.stencil ? GL_DEPTH_STENCIL : GL_DEPTH_COMPONENT;
        GLenum depthType = _desc.stencil ? GL_UNSIGNED_INT_24_8 : GL_FLOAT;
        glTexImage2D(GL_TEXTURE_2D, 0, depthFmt, w, h, 0, depthBase, depthType,
                     nullptr);
    }

    // Resize MSAA RBOs
    if (_desc.msaaSamples > 0) {
        glBindRenderbuffer(GL_RENDERBUFFER, _msaaColorRbo);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, _desc.msaaSamples,
                                         GL_RGB8, w, h);
        glBindRenderbuffer(GL_RENDERBUFFER, _msaaDepthRbo);
        GLenum msaaDepthFmt =
            _desc.stencil ? GL_DEPTH24_STENCIL8 : GL_DEPTH_COMPONENT32;
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, _desc.msaaSamples,
                                         msaaDepthFmt, w, h);
    }
}

void OpenGLFramebuffer::blitToScreen(int scrWidth, int scrHeight) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, _fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, _desc.width, _desc.height, 0, 0, scrWidth,
                      scrHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Texture* OpenGLFramebuffer::getColorTexture() {
    return nullptr;
} // TODO: wrap _colorTex
Texture* OpenGLFramebuffer::getDepthTexture() {
    return nullptr;
} // TODO: wrap _depthTex
Texture* OpenGLFramebuffer::getStencilTexture() { return nullptr; }
Texture* OpenGLFramebuffer::getDepthStencilTexture() { return nullptr; }

} // namespace Backend
} // namespace KE