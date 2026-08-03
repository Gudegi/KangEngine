///
/// Author Kyungwon Kang, 2024/11
///

#include "opengl_device.hpp"
#include "../base/base_utils.hpp"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <variant>
#include <glm/gtc/type_ptr.hpp>
#include <stb_image.h>

namespace KE {
namespace Backend {

namespace {

struct GLFramebufferColorFormat {
    GLenum internalFormat = GL_RGBA8;
    GLenum format = GL_RGBA;
    GLenum type = GL_UNSIGNED_BYTE;
};

struct GLTextureFormat {
    GLint internalFormat = GL_RGBA8;
    GLenum format = GL_RGBA;
    GLenum type = GL_UNSIGNED_BYTE;
    size_t bytesPerPixel = 4;
    bool depth = false;
    bool stencil = false;
};

GLTextureFormat toGLTextureFormat(TextureFormat format) {
    switch (format) {
    case TextureFormat::R8Unorm:
        return {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1, false, false};
    case TextureFormat::RGBA8Unorm:
        return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, false, false};
    case TextureFormat::RGBA8UnormSrgb:
        return {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, false, false};
    case TextureFormat::RGBA16Float:
        return {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, 8, false, false};
    case TextureFormat::Depth24Stencil8:
        return {GL_DEPTH24_STENCIL8,
                GL_DEPTH_STENCIL,
                GL_UNSIGNED_INT_24_8,
                4,
                true,
                true};
    case TextureFormat::Depth32Float:
        return {GL_DEPTH_COMPONENT32F,
                GL_DEPTH_COMPONENT,
                GL_FLOAT,
                4,
                true,
                false};
    case TextureFormat::Undefined:
        break;
    }
    throw std::invalid_argument("unsupported TextureFormat");
}

GLFramebufferColorFormat
toGLFramebufferColorFormat(FramebufferColorFormat format) {
    switch (format) {
    case FramebufferColorFormat::RGBA8:
        return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};
    case FramebufferColorFormat::RGBA16F:
        return {GL_RGBA16F, GL_RGBA, GL_FLOAT};
    }
    return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};
}

GLenum toGLTextureWrap(TextureWrap wrap) {
    switch (wrap) {
    case TextureWrap::Repeat:
        return GL_REPEAT;
    case TextureWrap::ClampToEdge:
        return GL_CLAMP_TO_EDGE;
    case TextureWrap::MirroredRepeat:
        return GL_MIRRORED_REPEAT;
    }
    return GL_REPEAT;
}

GLenum toGLTextureFilter(TextureFilter filter) {
    switch (filter) {
    case TextureFilter::Nearest:
        return GL_NEAREST;
    case TextureFilter::Linear:
        return GL_LINEAR;
    case TextureFilter::LinearMipmapLinear:
        return GL_LINEAR_MIPMAP_LINEAR;
    }
    return GL_LINEAR;
}

GLenum toGLTextureMagFilter(TextureFilter filter) {
    return filter == TextureFilter::Nearest ? GL_NEAREST : GL_LINEAR;
}

} // namespace

// OpenGLBuffer Implementation
OpenGLBuffer::OpenGLBuffer(BufferType type, size_t size, const void* data)
    : _type(type), _size(size) {

    switch (type) {
    case BufferType::Vertex:
        _usage = BufferUsage::Vertex;
        _target = GL_ARRAY_BUFFER;
        break;
    case BufferType::DynamicVertex:
        _usage = BufferUsage::Vertex | BufferUsage::CopyDst;
        _target = GL_ARRAY_BUFFER;
        break;
    case BufferType::Index:
        _usage = BufferUsage::Index;
        _target = GL_ELEMENT_ARRAY_BUFFER;
        break;
    case BufferType::Uniform:
        _usage = BufferUsage::Uniform | BufferUsage::CopyDst;
        _target = GL_UNIFORM_BUFFER;
        break;
    }

    GLenum usage =
        (type == BufferType::Uniform || type == BufferType::DynamicVertex)
            ? GL_DYNAMIC_DRAW
            : GL_STATIC_DRAW;
    glGenBuffers(1, &_buffer);
    glBindBuffer(_target, _buffer);
    glBufferData(_target, size, data, usage);
    glBindBuffer(_target, 0);
}

OpenGLBuffer::OpenGLBuffer(const BufferDesc& desc, const void* data)
    : _size(desc.size), _usage(desc.usage) {
    if (_size == 0 || _usage == BufferUsage::None)
        throw std::invalid_argument("portable buffer requires size and usage");
    if (hasFlag(_usage, BufferUsage::Vertex)) {
        _type = BufferType::Vertex;
        _target = GL_ARRAY_BUFFER;
    } else if (hasFlag(_usage, BufferUsage::Index)) {
        _type = BufferType::Index;
        _target = GL_ELEMENT_ARRAY_BUFFER;
    } else if (hasFlag(_usage, BufferUsage::Uniform)) {
        _type = BufferType::Uniform;
        _target = GL_UNIFORM_BUFFER;
    } else {
        throw std::invalid_argument(
            "portable OpenGL buffer requires Vertex, Index, or Uniform usage");
    }
    glGenBuffers(1, &_buffer);
    glBindBuffer(_target, _buffer);
    glBufferData(_target, static_cast<GLsizeiptr>(_size), data,
                 hasFlag(_usage, BufferUsage::CopyDst) ? GL_DYNAMIC_DRAW
                                                       : GL_STATIC_DRAW);
    glBindBuffer(_target, 0);
    if (glObjectLabel != nullptr && !desc.label.empty())
        glObjectLabel(GL_BUFFER, _buffer, -1, desc.label.c_str());
}

OpenGLBuffer::~OpenGLBuffer() {
#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
    if (_cudaResource)
        cudaGraphicsUnregisterResource(_cudaResource);
#endif
    glDeleteBuffers(1, &_buffer);
}

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

#ifdef KANGENGINE_USE_CUDA_GL_INTEROP
namespace {

void checkCudaInterop(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

} // namespace

cudaGraphicsResource* OpenGLBuffer::cudaResource() {
    if (!_cudaResource) {
        checkCudaInterop(
            cudaGraphicsGLRegisterBuffer(&_cudaResource, _buffer,
                                         cudaGraphicsRegisterFlagsWriteDiscard),
            "cudaGraphicsGLRegisterBuffer");
    }
    return _cudaResource;
}

bool OpenGLBuffer::setExternalData(const Sim::GpuArrayView& view, size_t count,
                                   size_t elementSize,
                                   size_t sourceStrideBytes) {
    if (!view.isCuda())
        return false;
    if (!view.data)
        throw std::runtime_error("CUDA external buffer has a null pointer");
    if (count * elementSize > _size)
        throw std::runtime_error(
            "CUDA external data exceeds the OpenGL buffer capacity");

    int previousDevice = 0;
    checkCudaInterop(cudaGetDevice(&previousDevice), "cudaGetDevice");
    if (view.deviceId >= 0 && view.deviceId != previousDevice)
        checkCudaInterop(cudaSetDevice(view.deviceId), "cudaSetDevice");

    cudaResource();

    auto stream = reinterpret_cast<cudaStream_t>(view.streamHandle);
    if (view.readyEventHandle != 0) {
        auto event = reinterpret_cast<cudaEvent_t>(view.readyEventHandle);
        checkCudaInterop(cudaStreamWaitEvent(stream, event, 0),
                         "cudaStreamWaitEvent");
    }

    checkCudaInterop(cudaGraphicsMapResources(1, &_cudaResource, stream),
                     "cudaGraphicsMapResources");

    void* destination = nullptr;
    size_t mappedSize = 0;
    cudaError_t mappedResult = cudaGraphicsResourceGetMappedPointer(
        &destination, &mappedSize, _cudaResource);
    if (mappedResult != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &_cudaResource, stream);
        checkCudaInterop(mappedResult, "cudaGraphicsResourceGetMappedPointer");
    }
    if (count * elementSize > mappedSize) {
        cudaGraphicsUnmapResources(1, &_cudaResource, stream);
        throw std::runtime_error(
            "Mapped OpenGL buffer is smaller than the CUDA source data");
    }

    const size_t sourcePitch =
        sourceStrideBytes == 0 ? elementSize : sourceStrideBytes;
    cudaError_t copyResult =
        cudaMemcpy2DAsync(destination, elementSize, view.data, sourcePitch,
                          elementSize, count, cudaMemcpyDeviceToDevice, stream);
    if (copyResult != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &_cudaResource, stream);
        checkCudaInterop(copyResult, "cudaMemcpy2DAsync");
    }

    checkCudaInterop(cudaGraphicsUnmapResources(1, &_cudaResource, stream),
                     "cudaGraphicsUnmapResources");
    if (view.deviceId >= 0 && view.deviceId != previousDevice)
        checkCudaInterop(cudaSetDevice(previousDevice), "cudaSetDevice");
    return true;
}

bool OpenGLDevice::mapCudaBuffers(const std::vector<Buffer*>& buffers,
                                  std::vector<Sim::GpuArrayView>& views,
                                  size_t count, size_t elementSize,
                                  int deviceId, uint64_t streamHandle) {
    views.clear();
    if (buffers.empty())
        return true;

    int previousDevice = 0;
    checkCudaInterop(cudaGetDevice(&previousDevice), "cudaGetDevice");
    if (deviceId >= 0 && deviceId != previousDevice)
        checkCudaInterop(cudaSetDevice(deviceId), "cudaSetDevice");

    std::vector<cudaGraphicsResource*> resources;
    resources.reserve(buffers.size());
    for (auto* buffer : buffers) {
        auto* glBuffer = dynamic_cast<OpenGLBuffer*>(buffer);
        if (!glBuffer || count * elementSize > glBuffer->size()) {
            if (deviceId >= 0 && deviceId != previousDevice)
                cudaSetDevice(previousDevice);
            return false;
        }
        resources.push_back(glBuffer->cudaResource());
    }

    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    checkCudaInterop(
        cudaGraphicsMapResources(static_cast<int>(resources.size()),
                                 resources.data(), stream),
        "cudaGraphicsMapResources(batch)");
    try {
        views.reserve(resources.size());
        for (auto* resource : resources) {
            void* pointer = nullptr;
            size_t mappedSize = 0;
            checkCudaInterop(cudaGraphicsResourceGetMappedPointer(
                                 &pointer, &mappedSize, resource),
                             "cudaGraphicsResourceGetMappedPointer(batch)");
            if (count * elementSize > mappedSize)
                throw std::runtime_error(
                    "mapped OpenGL transform buffer is too small");
            Sim::GpuArrayView view;
            view.data = pointer;
            view.memoryType = Sim::SimMemoryType::OpenGLBuffer;
            view.dtype = Sim::SimDType::Float32;
            view.lifetime = Sim::SimLifetimePolicy::Borrowed;
            view.deviceId = deviceId;
            view.shape = {static_cast<int64_t>(count), 4, 4};
            view.strides = {16, 4, 1};
            view.streamHandle = streamHandle;
            views.push_back(std::move(view));
        }
    } catch (...) {
        cudaGraphicsUnmapResources(static_cast<int>(resources.size()),
                                   resources.data(), stream);
        if (deviceId >= 0 && deviceId != previousDevice)
            cudaSetDevice(previousDevice);
        throw;
    }

    if (deviceId >= 0 && deviceId != previousDevice)
        checkCudaInterop(cudaSetDevice(previousDevice), "cudaSetDevice");
    return true;
}

void OpenGLDevice::unmapCudaBuffers(const std::vector<Buffer*>& buffers,
                                    int deviceId, uint64_t streamHandle) {
    if (buffers.empty())
        return;
    int previousDevice = 0;
    checkCudaInterop(cudaGetDevice(&previousDevice), "cudaGetDevice");
    if (deviceId >= 0 && deviceId != previousDevice)
        checkCudaInterop(cudaSetDevice(deviceId), "cudaSetDevice");

    std::vector<cudaGraphicsResource*> resources;
    resources.reserve(buffers.size());
    for (auto* buffer : buffers) {
        auto* glBuffer = dynamic_cast<OpenGLBuffer*>(buffer);
        if (!glBuffer)
            throw std::runtime_error(
                "CUDA transform unmap requires OpenGL buffers");
        resources.push_back(glBuffer->cudaResource());
    }
    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    checkCudaInterop(
        cudaGraphicsUnmapResources(static_cast<int>(resources.size()),
                                   resources.data(), stream),
        "cudaGraphicsUnmapResources(batch)");
    if (deviceId >= 0 && deviceId != previousDevice)
        checkCudaInterop(cudaSetDevice(previousDevice), "cudaSetDevice");
}
#endif

// OpenGLShader Implementation
OpenGLShader::OpenGLShader(const ShaderDesc& desc) : _desc(desc) {
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

    auto bindUniformBlockIfPresent = [&](const char* blockName, int slot) {
        GLuint blockIndex = glGetUniformBlockIndex(_shaderProgram, blockName);
        if (blockIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(_shaderProgram, blockIndex, slot);
    };
    bindUniformBlockIfPresent("cameraUBO", 0);
    bindUniformBlockIfPresent("lightUBO", 1);
    bindUniformBlockIfPresent("shadowUBO", 2);

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

void OpenGLShader::setMat4Array(const std::string& name,
                                const glm::mat4* values, size_t count) {
    if (!values || count == 0)
        return;
    glUniformMatrix4fv(glGetUniformLocation(_shaderProgram, name.c_str()),
                       static_cast<GLsizei>(count), GL_FALSE, &values[0][0][0]);
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
    : _width(desc.width), _height(desc.height), _channels(desc.channels),
      _format(desc.channels == 1 ? TextureFormat::R8Unorm
                                 : TextureFormat::RGBA8Unorm),
      _usage(TextureUsage::TextureBinding | TextureUsage::CopyDst) {

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

OpenGLTexture::OpenGLTexture(const TextureDesc& desc,
                             const SamplerDesc& sampler)
    : _width(desc.width), _height(desc.height), _channels(desc.channels),
      _format(desc.channels == 1 ? TextureFormat::R8Unorm
                                 : TextureFormat::RGBA8Unorm),
      _usage(TextureUsage::TextureBinding | TextureUsage::CopyDst) {

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

    setWrapParam(toGLTextureWrap(sampler.wrapU),
                 toGLTextureWrap(sampler.wrapV));
    setFilterParam(toGLTextureFilter(sampler.minFilter),
                   toGLTextureMagFilter(sampler.magFilter));

    glBindTexture(_target, 0);
}

OpenGLTexture::OpenGLTexture(const TextureDesc& desc, float warpParam,
                             float filterMinParam, float filterMaxParam)
    : _width(desc.width), _height(desc.height), _channels(desc.channels),
      _format(desc.channels == 1 ? TextureFormat::R8Unorm
                                 : TextureFormat::RGBA8Unorm),
      _usage(TextureUsage::TextureBinding | TextureUsage::CopyDst),
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

OpenGLTexture::OpenGLTexture(int w, int h, FramebufferColorFormat colorFormat)
    : _width(w), _height(h), _channels(4),
      _format(colorFormat == FramebufferColorFormat::RGBA16F
                  ? TextureFormat::RGBA16Float
                  : TextureFormat::RGBA8Unorm),
      _usage(TextureUsage::RenderAttachment | TextureUsage::TextureBinding |
             TextureUsage::CopySrc) {
    const GLFramebufferColorFormat glFormat =
        toGLFramebufferColorFormat(colorFormat);
    glGenTextures(1, &_textureID);
    glBindTexture(GL_TEXTURE_2D, _textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, glFormat.internalFormat, w, h, 0,
                 glFormat.format, glFormat.type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

OpenGLTexture::OpenGLTexture(int w, int h, bool stencil)
    : _width(w), _height(h), _channels(1),
      _format(stencil ? TextureFormat::Depth24Stencil8
                      : TextureFormat::Depth32Float),
      _usage(TextureUsage::RenderAttachment | TextureUsage::TextureBinding |
             TextureUsage::CopySrc) {
    GLenum internalFmt = stencil ? GL_DEPTH24_STENCIL8 : GL_DEPTH_COMPONENT32;
    GLenum baseFmt = stencil ? GL_DEPTH_STENCIL : GL_DEPTH_COMPONENT;
    GLenum dataType = stencil ? GL_UNSIGNED_INT_24_8 : GL_FLOAT;
    glGenTextures(1, &_textureID);
    glBindTexture(GL_TEXTURE_2D, _textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFmt, w, h, 0, baseFmt, dataType,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Swizzle R>RGB: ImGui::Image displays as grayscale.
    // PCF only reads .r (= GL_RED = depth), unaffected.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ONE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

OpenGLTexture::OpenGLTexture(const TextureResourceDesc& desc,
                             const TextureInitialData* initialData)
    : _target(desc.sampleCount > 1 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D),
      _width(static_cast<int>(desc.extent.width)),
      _height(static_cast<int>(desc.extent.height)), _channels(0),
      _format(desc.format), _usage(desc.usage),
      _mipLevelCount(desc.mipLevelCount), _sampleCount(desc.sampleCount),
      _depthOrArrayLayers(desc.extent.depthOrArrayLayers),
      _dimension(desc.dimension),
      _portableResource(true) {
    const DescriptorValidationResult validation = validate(desc);
    if (!validation)
        throw std::invalid_argument(
            std::string("invalid texture descriptor: ") + validation.message);
    if (desc.dimension != TextureDimension::D2 ||
        desc.extent.depthOrArrayLayers != 1)
        throw std::invalid_argument("OpenGL bootstrap currently supports only "
                                    "single-layer 2D textures");
    if (initialData && desc.sampleCount > 1)
        throw std::invalid_argument(
            "multisampled textures do not accept initial data");

    const GLTextureFormat glFormat = toGLTextureFormat(desc.format);
    size_t initialBytesPerRow = 0;
    if (initialData) {
        const size_t tightBytesPerRow =
            static_cast<size_t>(_width) * glFormat.bytesPerPixel;
        initialBytesPerRow = initialData->bytesPerRow ? initialData->bytesPerRow
                                                      : tightBytesPerRow;
        if (!initialData->data)
            throw std::invalid_argument("initial texture data is null");
        if (initialBytesPerRow < tightBytesPerRow ||
            initialBytesPerRow % glFormat.bytesPerPixel != 0)
            throw std::invalid_argument(
                "initial texture bytesPerRow is incompatible with format");
        const size_t requiredSize =
            initialBytesPerRow * static_cast<size_t>(_height - 1) +
            tightBytesPerRow;
        if (initialData->size < requiredSize)
            throw std::invalid_argument(
                "initial texture data is smaller than the declared extent");
    }

    glGenTextures(1, &_textureID);
    glBindTexture(_target, _textureID);

    if (desc.sampleCount > 1) {
        glTexImage2DMultisample(_target, static_cast<GLsizei>(desc.sampleCount),
                                glFormat.internalFormat, _width, _height,
                                GL_TRUE);
    } else {
        glTexParameteri(_target, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(_target, GL_TEXTURE_MAX_LEVEL,
                        static_cast<GLint>(desc.mipLevelCount - 1));
        const void* levelZeroData = initialData ? initialData->data : nullptr;
        if (initialData) {
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glPixelStorei(GL_UNPACK_ROW_LENGTH,
                          static_cast<GLint>(initialBytesPerRow /
                                             glFormat.bytesPerPixel));
        }

        int mipWidth = _width;
        int mipHeight = _height;
        for (uint32_t level = 0; level < desc.mipLevelCount; ++level) {
            glTexImage2D(_target, static_cast<GLint>(level),
                         glFormat.internalFormat, mipWidth, mipHeight, 0,
                         glFormat.format, glFormat.type,
                         level == 0 ? levelZeroData : nullptr);
            mipWidth = std::max(1, mipWidth / 2);
            mipHeight = std::max(1, mipHeight / 2);
        }

        if (initialData) {
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        }
    }

    if (glObjectLabel != nullptr && !desc.label.empty())
        glObjectLabel(GL_TEXTURE, _textureID, -1, desc.label.c_str());
    glBindTexture(_target, 0);
}

OpenGLTexture::OpenGLTexture(GLuint cubemapHandle, int faceWidth,
                             int faceHeight)
    : _textureID(cubemapHandle), _target(GL_TEXTURE_CUBE_MAP),
      _width(faceWidth), _height(faceHeight), _channels(4),
      _format(TextureFormat::RGBA8Unorm),
      _usage(TextureUsage::TextureBinding), _mipLevelCount(1),
      _sampleCount(1), _depthOrArrayLayers(6),
      _dimension(TextureDimension::D2), _portableResource(true) {
    if (!_textureID || faceWidth <= 0 || faceHeight <= 0)
        throw std::invalid_argument("invalid OpenGL cubemap texture");
}

OpenGLTexture::~OpenGLTexture() { glDeleteTextures(1, &_textureID); }

void OpenGLTexture::bind(int slot) {
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(_target, _textureID);
}

void OpenGLTexture::unbind() { glBindTexture(_target, 0); }

void OpenGLTexture::setWrapParam(GLenum wrapU, GLenum wrapV) const {
    glTexParameteri(_target, GL_TEXTURE_WRAP_S, wrapU);
    glTexParameteri(_target, GL_TEXTURE_WRAP_T, wrapV);
}

void OpenGLTexture::setWarpParam(GLfloat warpParam) const {
    glTexParameteri(_target, GL_TEXTURE_WRAP_S, warpParam);
    glTexParameteri(_target, GL_TEXTURE_WRAP_T, warpParam);
}

void OpenGLTexture::setFilterParam(GLfloat filterMinParam,
                                   GLfloat filterMaxParam) const {
    glTexParameteri(_target, GL_TEXTURE_MIN_FILTER, filterMinParam);
    glTexParameteri(_target, GL_TEXTURE_MAG_FILTER, filterMaxParam);
}

OpenGLTextureView::OpenGLTextureView(OpenGLTexture* texture,
                                     TextureViewDesc desc)
    : _texture(texture), _desc(std::move(desc)) {
    if (!_texture)
        throw std::invalid_argument("texture view requires a texture");
    if (_desc.format == TextureFormat::Undefined)
        _desc.format = _texture->format();

    TextureResourceDesc textureDesc;
    textureDesc.extent = {
        static_cast<uint32_t>(_texture->getWidth()),
        static_cast<uint32_t>(_texture->getHeight()),
        _texture->getDepthOrArrayLayers()};
    textureDesc.dimension = _texture->getDimension();
    textureDesc.format = _texture->format();
    textureDesc.usage = _texture->usage();
    textureDesc.mipLevelCount = _texture->mipLevelCount();
    textureDesc.sampleCount = _texture->sampleCount();
    const DescriptorValidationResult validation = validate(_desc, textureDesc);
    if (!validation)
        throw std::invalid_argument(std::string("invalid texture view: ") +
                                    validation.message);
    if (_desc.baseMipLevel != 0 ||
        _desc.mipLevelCount != _texture->mipLevelCount() ||
        _desc.baseArrayLayer != 0 ||
        _desc.arrayLayerCount != _texture->getDepthOrArrayLayers())
        throw std::invalid_argument(
            "OpenGL bootstrap supports full-resource texture views only");
}

OpenGLSampler::OpenGLSampler(const SamplerDesc& desc) {
    glGenSamplers(1, &_sampler);
    glSamplerParameteri(_sampler, GL_TEXTURE_WRAP_S,
                        toGLTextureWrap(desc.wrapU));
    glSamplerParameteri(_sampler, GL_TEXTURE_WRAP_T,
                        toGLTextureWrap(desc.wrapV));
    glSamplerParameteri(_sampler, GL_TEXTURE_MIN_FILTER,
                        toGLTextureFilter(desc.minFilter));
    glSamplerParameteri(_sampler, GL_TEXTURE_MAG_FILTER,
                        toGLTextureMagFilter(desc.magFilter));
    if (glObjectLabel != nullptr && !desc.label.empty())
        glObjectLabel(GL_SAMPLER, _sampler, -1, desc.label.c_str());
}

OpenGLSampler::~OpenGLSampler() {
    if (_sampler)
        glDeleteSamplers(1, &_sampler);
}

OpenGLRenderTarget::OpenGLRenderTarget(const RenderPassDesc& desc)
    : _desc(desc) {
    if (_desc.colorAttachments.empty() && !_desc.depthStencilAttachment)
        throw std::invalid_argument(
            "render target requires at least one attachment");

    GLint maxColorAttachments = 0;
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxColorAttachments);
    if (_desc.colorAttachments.size() >
        static_cast<size_t>(maxColorAttachments))
        throw std::invalid_argument(
            "render target exceeds GL_MAX_COLOR_ATTACHMENTS");

    auto validateCommon = [&](OpenGLTextureView* view) {
        if (!view)
            throw std::invalid_argument("render attachment view is null");
        auto* texture = dynamic_cast<OpenGLTexture*>(view->getTexture());
        if (!texture)
            throw std::invalid_argument(
                "render attachment must reference an OpenGL texture");
        if (!hasFlag(texture->usage(), TextureUsage::RenderAttachment))
            throw std::invalid_argument(
                "render attachment texture is missing RenderAttachment usage");
        if (_width == 0) {
            _width = texture->getWidth();
            _height = texture->getHeight();
            _sampleCount = texture->sampleCount();
        } else if (_width != texture->getWidth() ||
                   _height != texture->getHeight() ||
                   _sampleCount != texture->sampleCount()) {
            throw std::invalid_argument("render attachments must have matching "
                                        "extent and sample count");
        }
        return texture;
    };

    glGenFramebuffers(1, &_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    try {
        _drawBuffers.reserve(_desc.colorAttachments.size());
        _resolveTextures.resize(_desc.colorAttachments.size(), nullptr);
        std::vector<GLuint> attachedColorTextures;
        attachedColorTextures.reserve(_desc.colorAttachments.size());
        for (size_t index = 0; index < _desc.colorAttachments.size(); ++index) {
            const ColorAttachmentDesc& attachment =
                _desc.colorAttachments[index];
            if (attachment.storeOp == StoreOp::Discard)
                throw std::invalid_argument(
                    "StoreOp::Discard is not implemented by the OpenGL "
                    "bootstrap");
            auto* view = dynamic_cast<OpenGLTextureView*>(attachment.view);
            OpenGLTexture* texture = validateCommon(view);
            if (isDepthFormat(texture->format()) ||
                view->getDesc().aspect != TextureAspect::All)
                throw std::invalid_argument("color attachment requires an "
                                            "all-aspect color texture view");
            if (std::find(attachedColorTextures.begin(),
                          attachedColorTextures.end(),
                          texture->getHandle()) != attachedColorTextures.end())
                throw std::invalid_argument(
                    "MRT color attachments must reference distinct textures");
            attachedColorTextures.push_back(texture->getHandle());

            if (attachment.resolveTarget) {
                auto* resolveView = dynamic_cast<OpenGLTextureView*>(
                    attachment.resolveTarget);
                auto* resolveTexture = resolveView
                                           ? dynamic_cast<OpenGLTexture*>(
                                                 resolveView->getTexture())
                                           : nullptr;
                if (!resolveTexture ||
                    !hasFlag(resolveTexture->usage(),
                             TextureUsage::RenderAttachment) ||
                    isDepthFormat(resolveTexture->format()) ||
                    resolveView->getDesc().aspect != TextureAspect::All)
                    throw std::invalid_argument(
                        "resolve target requires a color render attachment");
                if (texture->sampleCount() <= 1 ||
                    resolveTexture->sampleCount() != 1)
                    throw std::invalid_argument(
                        "resolve requires multisampled source and single-sample target");
                if (resolveTexture->getWidth() != texture->getWidth() ||
                    resolveTexture->getHeight() != texture->getHeight() ||
                    resolveTexture->format() != texture->format())
                    throw std::invalid_argument(
                        "resolve source and target must match extent and format");
                if (resolveTexture->getHandle() == texture->getHandle())
                    throw std::invalid_argument(
                        "resolve source and target must be distinct");
                _resolveTextures[index] = resolveTexture;
            }

            const GLenum point =
                GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(index);
            glFramebufferTexture2D(GL_FRAMEBUFFER, point,
                                   texture->sampleCount() > 1
                                       ? GL_TEXTURE_2D_MULTISAMPLE
                                       : GL_TEXTURE_2D,
                                   texture->getHandle(), 0);
            _drawBuffers.push_back(point);
        }

        if (_drawBuffers.empty()) {
            glDrawBuffer(GL_NONE);
            glReadBuffer(GL_NONE);
        } else {
            glDrawBuffers(static_cast<GLsizei>(_drawBuffers.size()),
                          _drawBuffers.data());
        }

        if (_desc.depthStencilAttachment) {
            const DepthStencilAttachmentDesc& attachment =
                *_desc.depthStencilAttachment;
            if (attachment.depthStoreOp == StoreOp::Discard ||
                attachment.stencilStoreOp == StoreOp::Discard)
                throw std::invalid_argument(
                    "StoreOp::Discard is not implemented by the OpenGL "
                    "bootstrap");
            auto* view = dynamic_cast<OpenGLTextureView*>(attachment.view);
            OpenGLTexture* texture = validateCommon(view);
            if (!isDepthFormat(texture->format()) ||
                view->getDesc().aspect == TextureAspect::StencilOnly)
                throw std::invalid_argument(
                    "depth attachment requires a depth-capable texture view");
            const GLenum point = hasStencilAspect(texture->format())
                                     ? GL_DEPTH_STENCIL_ATTACHMENT
                                     : GL_DEPTH_ATTACHMENT;
            glFramebufferTexture2D(GL_FRAMEBUFFER, point,
                                   texture->sampleCount() > 1
                                       ? GL_TEXTURE_2D_MULTISAMPLE
                                       : GL_TEXTURE_2D,
                                   texture->getHandle(), 0);
        }

        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            std::string attachments;
            for (size_t index = 0; index < _desc.colorAttachments.size();
                 ++index) {
                const auto* view = static_cast<const OpenGLTextureView*>(
                    _desc.colorAttachments[index].view);
                attachments += " color[" + std::to_string(index) + "]='" +
                               view->getDesc().label + "' format=" +
                               std::to_string(static_cast<uint32_t>(
                                   view->getTexture()->getFormat()));
            }
            if (_desc.depthStencilAttachment) {
                const auto* view = static_cast<const OpenGLTextureView*>(
                    _desc.depthStencilAttachment->view);
                attachments +=
                    " depth='" + view->getDesc().label + "' format=" +
                    std::to_string(
                        static_cast<uint32_t>(view->getTexture()->getFormat()));
            }
            throw std::runtime_error(
                "OpenGL render target '" + _desc.label +
                "' is incomplete (status=" +
                std::to_string(static_cast<uint32_t>(status)) +
                "):" + attachments);
        }

        if (glObjectLabel != nullptr && !_desc.label.empty())
            glObjectLabel(GL_FRAMEBUFFER, _fbo, -1, _desc.label.c_str());
        if (std::any_of(_resolveTextures.begin(), _resolveTextures.end(),
                        [](const OpenGLTexture* texture) {
                            return texture != nullptr;
                        })) {
            glGenFramebuffers(1, &_resolveFbo);
            if (glObjectLabel != nullptr && !_desc.label.empty())
                glObjectLabel(GL_FRAMEBUFFER, _resolveFbo, -1,
                              (_desc.label + "_resolve").c_str());
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    } catch (...) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &_fbo);
        if (_resolveFbo)
            glDeleteFramebuffers(1, &_resolveFbo);
        _fbo = 0;
        _resolveFbo = 0;
        throw;
    }
}

OpenGLRenderTarget::~OpenGLRenderTarget() {
    if (_fbo)
        glDeleteFramebuffers(1, &_fbo);
    if (_resolveFbo)
        glDeleteFramebuffers(1, &_resolveFbo);
}

void OpenGLRenderTarget::beginPass() {
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    if (_drawBuffers.empty()) {
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
    } else {
        glDrawBuffers(static_cast<GLsizei>(_drawBuffers.size()),
                      _drawBuffers.data());
    }

    for (size_t index = 0; index < _desc.colorAttachments.size(); ++index) {
        const ColorAttachmentDesc& attachment = _desc.colorAttachments[index];
        if (attachment.loadOp == LoadOp::Clear) {
            const GLfloat value[4] = {
                attachment.clearValue.r, attachment.clearValue.g,
                attachment.clearValue.b, attachment.clearValue.a};
            glClearBufferfv(GL_COLOR, static_cast<GLint>(index), value);
        }
    }

    if (_desc.depthStencilAttachment) {
        const DepthStencilAttachmentDesc& attachment =
            *_desc.depthStencilAttachment;
        auto* view = static_cast<OpenGLTextureView*>(attachment.view);
        auto* texture = static_cast<OpenGLTexture*>(view->getTexture());
        const bool clearDepth = attachment.depthLoadOp == LoadOp::Clear;
        const bool clearStencil = hasStencilAspect(texture->format()) &&
                                  attachment.stencilLoadOp == LoadOp::Clear;
        if (clearDepth && clearStencil) {
            glClearBufferfi(GL_DEPTH_STENCIL, 0, attachment.depthClearValue,
                            static_cast<GLint>(attachment.stencilClearValue));
        } else if (clearDepth) {
            const GLfloat depth = attachment.depthClearValue;
            glClearBufferfv(GL_DEPTH, 0, &depth);
        } else if (clearStencil) {
            const GLint stencil =
                static_cast<GLint>(attachment.stencilClearValue);
            glClearBufferiv(GL_STENCIL, 0, &stencil);
        }
    }
}

void OpenGLRenderTarget::endPass() {
    if (_resolveFbo) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, _fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _resolveFbo);
        for (size_t index = 0; index < _resolveTextures.size(); ++index) {
            OpenGLTexture* resolve = _resolveTextures[index];
            if (!resolve)
                continue;
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, resolve->getHandle(), 0);
            glReadBuffer(GL_COLOR_ATTACHMENT0 + static_cast<GLenum>(index));
            glDrawBuffer(GL_COLOR_ATTACHMENT0);
            if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) !=
                GL_FRAMEBUFFER_COMPLETE)
                throw std::runtime_error(
                    "OpenGL resolve framebuffer is incomplete");
            glBlitFramebuffer(0, 0, _width, _height, 0, 0, _width, _height,
                              GL_COLOR_BUFFER_BIT, GL_NEAREST);
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

OpenGLBindGroupLayout::OpenGLBindGroupLayout(BindGroupLayoutDesc desc)
    : _desc(std::move(desc)) {
    std::vector<uint32_t> bindings;
    uint32_t samplerCount = 0;
    for (const auto& entry : _desc.entries) {
        if (entry.visibility == ShaderStageVisibility::None)
            throw std::invalid_argument("bind-group entry visibility is empty");
        if (std::find(bindings.begin(), bindings.end(), entry.binding) !=
            bindings.end())
            throw std::invalid_argument("duplicate bind-group layout binding");
        bindings.push_back(entry.binding);
        if (entry.type == BindingType::Sampler)
            ++samplerCount;
    }
    if (samplerCount > 1)
        throw std::invalid_argument(
            "OpenGL bootstrap supports one shared sampler per bind group");
}

OpenGLPipelineLayout::OpenGLPipelineLayout(PipelineLayoutDesc desc)
    : _desc(std::move(desc)) {
    for (auto* layout : _desc.bindGroupLayouts)
        if (!dynamic_cast<OpenGLBindGroupLayout*>(layout))
            throw std::invalid_argument(
                "pipeline layout contains a foreign/null layout");
}

OpenGLBindGroup::OpenGLBindGroup(BindGroupDesc desc) : _desc(std::move(desc)) {
    auto* layout = dynamic_cast<OpenGLBindGroupLayout*>(_desc.layout);
    if (!layout)
        throw std::invalid_argument("bind group requires an OpenGL layout");
    if (_desc.entries.size() != layout->getDesc().entries.size())
        throw std::invalid_argument(
            "bind group entry count does not match layout");
    for (const auto& layoutEntry : layout->getDesc().entries) {
        const auto it =
            std::find_if(_desc.entries.begin(), _desc.entries.end(),
                         [&](const BindGroupEntry& entry) {
                             return entry.binding == layoutEntry.binding;
                         });
        if (it == _desc.entries.end())
            throw std::invalid_argument(
                "bind group is missing a required binding");
        if (layoutEntry.type == BindingType::UniformBuffer) {
            if (!it->buffer ||
                !hasFlag(it->buffer->getUsage(), BufferUsage::Uniform))
                throw std::invalid_argument(
                    "uniform binding needs Uniform usage");
            const uint64_t size =
                it->size == 0 ? it->buffer->getSize() - it->offset : it->size;
            if (it->offset + size > it->buffer->getSize())
                throw std::invalid_argument(
                    "uniform binding range is out of bounds");
        } else if (layoutEntry.type == BindingType::SampledTexture) {
            const bool expectsDepth =
                layoutEntry.textureSampleType == TextureSampleType::Depth;
            if (!it->textureView ||
                !hasFlag(it->textureView->getTexture()->getUsage(),
                         TextureUsage::TextureBinding) ||
                isDepthFormat(it->textureView->getTexture()->getFormat()) !=
                    expectsDepth ||
                it->textureView->getDesc().dimension !=
                    layoutEntry.textureViewDimension ||
                (layoutEntry.textureFormat != TextureFormat::Undefined &&
                 it->textureView->getTexture()->getFormat() !=
                     layoutEntry.textureFormat))
                throw std::invalid_argument("sampled texture binding mismatch");
        } else if (!it->sampler) {
            throw std::invalid_argument("sampler binding is null");
        }
    }
}

OpenGLGraphicsPipeline::OpenGLGraphicsPipeline(const GraphicsPipelineDesc& desc)
    : _desc(desc) {
    if (_desc.sampleCount != 1 && _desc.sampleCount != 4)
        throw std::invalid_argument(
            "OpenGL pipeline sampleCount must be 1 or 4");
    if (_desc.raster.depthClamp)
        throw std::invalid_argument("pipeline depth clamp is not implemented");
    for (const ShaderStage& stage : _desc.shader.stages) {
        if (stage.entryPoint != "main")
            throw std::invalid_argument(
                "OpenGL GLSL pipeline entry point must currently be 'main'");
    }
    for (const VertexBufferLayout& layout : _desc.vertexBuffers) {
        if (!layout.attributes.empty() && layout.arrayStride == 0)
            throw std::invalid_argument(
                "vertex buffer layout with attributes requires a stride");
        std::vector<uint32_t> locations;
        for (const VertexAttributeDesc& attribute : layout.attributes) {
            if (attribute.offset >= layout.arrayStride)
                throw std::invalid_argument(
                    "vertex attribute offset exceeds its buffer stride");
            if (std::find(locations.begin(), locations.end(),
                          attribute.shaderLocation) != locations.end())
                throw std::invalid_argument(
                    "duplicate shader location in vertex buffer layout");
            locations.push_back(attribute.shaderLocation);
        }
    }
    std::optional<ColorWriteMask> commonWriteMask;
    for (const ColorTargetState& target : _desc.colorTargets) {
        if (target.format == TextureFormat::Undefined ||
            isDepthFormat(target.format))
            throw std::invalid_argument("invalid pipeline color target format");
        if (commonWriteMask && *commonWriteMask != target.writeMask)
            throw std::invalid_argument(
                "distinct MRT write masks are deferred until indexed GL state");
        commonWriteMask = target.writeMask;
    }
    if (_desc.depthStencil && !isDepthFormat(_desc.depthStencil->format))
        throw std::invalid_argument("invalid pipeline depth format");
    if (_desc.depthStencil && _desc.depthStencil->depthBiasClamp != 0.0f)
        throw std::invalid_argument(
            "OpenGL 4.1 backend does not support depth-bias clamp");
    _shader = std::make_unique<OpenGLShader>(_desc.shader);
    glGenVertexArrays(1, &_vao);
    if (_desc.pipelineLayout) {
        auto* pipelineLayout =
            dynamic_cast<OpenGLPipelineLayout*>(_desc.pipelineLayout);
        if (!pipelineLayout)
            throw std::invalid_argument("pipeline uses a foreign layout");
        GLuint nextTextureSlot = 0;
        GLuint nextUniformSlot = 0;
        for (uint32_t groupIndex = 0;
             groupIndex < pipelineLayout->getDesc().bindGroupLayouts.size();
             ++groupIndex) {
            const auto& groupDesc = pipelineLayout->getDesc()
                                        .bindGroupLayouts[groupIndex]
                                        ->getDesc();
            for (const auto& entry : groupDesc.entries) {
                CachedBinding cached;
                cached.group = groupIndex;
                cached.binding = entry.binding;
                cached.type = entry.type;
                const std::string name = "ke_g" + std::to_string(groupIndex) +
                                         "_b" + std::to_string(entry.binding);
                if (entry.type == BindingType::SampledTexture) {
                    cached.location = glGetUniformLocation(_shader->getHandle(),
                                                           name.c_str());
                    cached.slot = nextTextureSlot++;
                } else if (entry.type == BindingType::UniformBuffer) {
                    const GLuint block = glGetUniformBlockIndex(
                        _shader->getHandle(), name.c_str());
                    cached.location = block == GL_INVALID_INDEX
                                          ? -1
                                          : static_cast<GLint>(block);
                    cached.slot = nextUniformSlot++;
                    if (cached.location >= 0)
                        glUniformBlockBinding(_shader->getHandle(), block,
                                              cached.slot);
                }
                _bindings.push_back(cached);
            }
        }
    }
}

OpenGLGraphicsPipeline::~OpenGLGraphicsPipeline() {
    if (_vao)
        glDeleteVertexArrays(1, &_vao);
}

void OpenGLGraphicsPipeline::apply() const {
    _shader->bind();
    glBindVertexArray(_vao);
    if (_desc.depthStencil) {
        glEnable(GL_DEPTH_TEST);
        glDepthMask(_desc.depthStencil->depthWriteEnabled ? GL_TRUE : GL_FALSE);
        GLenum compare = GL_LESS;
        switch (_desc.depthStencil->depthCompare) {
        case CompareFunction::Never:
            compare = GL_NEVER;
            break;
        case CompareFunction::Less:
            compare = GL_LESS;
            break;
        case CompareFunction::LessEqual:
            compare = GL_LEQUAL;
            break;
        case CompareFunction::Greater:
            compare = GL_GREATER;
            break;
        case CompareFunction::GreaterEqual:
            compare = GL_GEQUAL;
            break;
        case CompareFunction::Equal:
            compare = GL_EQUAL;
            break;
        case CompareFunction::NotEqual:
            compare = GL_NOTEQUAL;
            break;
        case CompareFunction::Always:
            compare = GL_ALWAYS;
            break;
        }
        glDepthFunc(compare);
        const bool depthBiasEnabled =
            _desc.depthStencil->depthBias != 0 ||
            _desc.depthStencil->depthBiasSlopeScale != 0.0f;
        if (depthBiasEnabled) {
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(_desc.depthStencil->depthBiasSlopeScale,
                            static_cast<float>(
                                _desc.depthStencil->depthBias));
        } else {
            glDisable(GL_POLYGON_OFFSET_FILL);
        }
    } else {
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glDisable(GL_POLYGON_OFFSET_FILL);
    }
    if (_desc.primitive.cullMode == CullMode::None) {
        glDisable(GL_CULL_FACE);
    } else {
        glEnable(GL_CULL_FACE);
        glCullFace(_desc.primitive.cullMode == CullMode::Front ? GL_FRONT
                                                               : GL_BACK);
    }
    glFrontFace(_desc.primitive.frontFace == FrontFace::CCW ? GL_CCW : GL_CW);
    if (!_desc.colorTargets.empty() && _desc.colorTargets.front().blend) {
        auto toFactor = [](BlendFactorValue factor) {
            switch (factor) {
            case BlendFactorValue::Zero: return GL_ZERO;
            case BlendFactorValue::One: return GL_ONE;
            case BlendFactorValue::Src: return GL_SRC_COLOR;
            case BlendFactorValue::OneMinusSrc: return GL_ONE_MINUS_SRC_COLOR;
            case BlendFactorValue::SrcAlpha: return GL_SRC_ALPHA;
            case BlendFactorValue::OneMinusSrcAlpha: return GL_ONE_MINUS_SRC_ALPHA;
            case BlendFactorValue::Dst: return GL_DST_COLOR;
            case BlendFactorValue::OneMinusDst: return GL_ONE_MINUS_DST_COLOR;
            case BlendFactorValue::DstAlpha: return GL_DST_ALPHA;
            case BlendFactorValue::OneMinusDstAlpha: return GL_ONE_MINUS_DST_ALPHA;
            }
            return GL_ONE;
        };
        auto toOperation = [](BlendOperation operation) {
            switch (operation) {
            case BlendOperation::Add: return GL_FUNC_ADD;
            case BlendOperation::Subtract: return GL_FUNC_SUBTRACT;
            case BlendOperation::ReverseSubtract: return GL_FUNC_REVERSE_SUBTRACT;
            case BlendOperation::Min: return GL_MIN;
            case BlendOperation::Max: return GL_MAX;
            }
            return GL_FUNC_ADD;
        };
        const BlendState& blend = *_desc.colorTargets.front().blend;
        glEnable(GL_BLEND);
        glBlendEquationSeparate(toOperation(blend.color.operation),
                                toOperation(blend.alpha.operation));
        glBlendFuncSeparate(toFactor(blend.color.srcFactor),
                            toFactor(blend.color.dstFactor),
                            toFactor(blend.alpha.srcFactor),
                            toFactor(blend.alpha.dstFactor));
    } else {
        glDisable(GL_BLEND);
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    const ColorWriteMask mask = _desc.colorTargets.empty()
                                    ? ColorWriteMask::None
                                    : _desc.colorTargets.front().writeMask;
    glColorMask(hasFlag(mask, ColorWriteMask::Red),
                hasFlag(mask, ColorWriteMask::Green),
                hasFlag(mask, ColorWriteMask::Blue),
                hasFlag(mask, ColorWriteMask::Alpha));
}

void OpenGLGraphicsPipeline::bindVertexBuffer(uint32_t slot,
                                              OpenGLBuffer* buffer,
                                              uint64_t offset) const {
    if (!buffer || !hasFlag(buffer->getUsage(), BufferUsage::Vertex))
        throw std::invalid_argument("vertex buffer is missing Vertex usage");
    if (slot >= _desc.vertexBuffers.size())
        throw std::invalid_argument(
            "vertex buffer slot is outside pipeline layout");
    const VertexBufferLayout& layout = _desc.vertexBuffers[slot];
    if (offset >= buffer->getSize())
        throw std::invalid_argument("vertex buffer offset exceeds buffer size");
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, buffer->getHandle());
    for (const VertexAttributeDesc& attribute : layout.attributes) {
        GLint components = 1;
        bool integer = false;
        switch (attribute.format) {
        case VertexFormat::Float32:
            components = 1;
            break;
        case VertexFormat::Float32x2:
            components = 2;
            break;
        case VertexFormat::Float32x3:
            components = 3;
            break;
        case VertexFormat::Float32x4:
            components = 4;
            break;
        case VertexFormat::Sint32x4:
            components = 4;
            integer = true;
            break;
        }
        glEnableVertexAttribArray(attribute.shaderLocation);
        if (integer) {
            glVertexAttribIPointer(
                attribute.shaderLocation, components, GL_INT,
                static_cast<GLsizei>(layout.arrayStride),
                reinterpret_cast<const void*>(offset + attribute.offset));
        } else {
            glVertexAttribPointer(
                attribute.shaderLocation, components, GL_FLOAT, GL_FALSE,
                static_cast<GLsizei>(layout.arrayStride),
                reinterpret_cast<const void*>(offset + attribute.offset));
        }
        glVertexAttribDivisor(attribute.shaderLocation,
                              layout.stepMode == VertexStepMode::Instance ? 1
                                                                          : 0);
    }
}

void OpenGLGraphicsPipeline::bindIndexBuffer(OpenGLBuffer* buffer) const {
    if (!buffer || !hasFlag(buffer->getUsage(), BufferUsage::Index))
        throw std::invalid_argument("index buffer is missing Index usage");
    glBindVertexArray(_vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer->getHandle());
}

void OpenGLGraphicsPipeline::bindGroup(uint32_t index,
                                       OpenGLBindGroup* group) const {
    if (!_desc.pipelineLayout || !group ||
        index >= _desc.pipelineLayout->getDesc().bindGroupLayouts.size() ||
        group->getDesc().layout !=
            _desc.pipelineLayout->getDesc().bindGroupLayouts[index])
        throw std::invalid_argument(
            "bind group is incompatible with pipeline layout");
    const auto& layoutEntries = group->getDesc().layout->getDesc().entries;
    for (const BindGroupEntry& entry : group->getDesc().entries) {
        const auto layoutIt =
            std::find_if(layoutEntries.begin(), layoutEntries.end(),
                         [&](const BindGroupLayoutEntry& item) {
                             return item.binding == entry.binding;
                         });
        const auto cacheIt = std::find_if(
            _bindings.begin(), _bindings.end(), [&](const CachedBinding& item) {
                return item.group == index && item.binding == entry.binding;
            });
        if (layoutIt->type == BindingType::SampledTexture) {
            auto* view = dynamic_cast<OpenGLTextureView*>(entry.textureView);
            glActiveTexture(GL_TEXTURE0 + cacheIt->slot);
            glBindTexture(view->getTarget(), view->getHandle());
            if (cacheIt->location >= 0)
                glUniform1i(cacheIt->location,
                            static_cast<GLint>(cacheIt->slot));
        } else if (layoutIt->type == BindingType::UniformBuffer) {
            auto* buffer = dynamic_cast<OpenGLBuffer*>(entry.buffer);
            const uint64_t size =
                entry.size == 0 ? buffer->getSize() - entry.offset : entry.size;
            glBindBufferRange(GL_UNIFORM_BUFFER, cacheIt->slot,
                              buffer->getHandle(), entry.offset, size);
        } else {
            auto* sampler = dynamic_cast<OpenGLSampler*>(entry.sampler);
            for (const CachedBinding& sampled : _bindings)
                if (sampled.group == index &&
                    sampled.type == BindingType::SampledTexture)
                    glBindSampler(sampled.slot, sampler->getHandle());
        }
    }
}

std::vector<GLuint> OpenGLGraphicsPipeline::textureSlots() const {
    std::vector<GLuint> result;
    for (const auto& binding : _bindings)
        if (binding.type == BindingType::SampledTexture &&
            std::find(result.begin(), result.end(), binding.slot) ==
                result.end())
            result.push_back(binding.slot);
    return result;
}

std::vector<GLuint> OpenGLGraphicsPipeline::uniformSlots() const {
    std::vector<GLuint> result;
    for (const auto& binding : _bindings)
        if (binding.type == BindingType::UniformBuffer &&
            std::find(result.begin(), result.end(), binding.slot) ==
                result.end())
            result.push_back(binding.slot);
    return result;
}

namespace {

GLenum toGLTopology(PrimitiveTopology topology) {
    switch (topology) {
    case PrimitiveTopology::PointList:
        return GL_POINTS;
    case PrimitiveTopology::LineList:
        return GL_LINES;
    case PrimitiveTopology::LineStrip:
        return GL_LINE_STRIP;
    case PrimitiveTopology::TriangleList:
        return GL_TRIANGLES;
    case PrimitiveTopology::TriangleStrip:
        return GL_TRIANGLE_STRIP;
    }
    return GL_TRIANGLES;
}

struct BeginPassCommand {
    OpenGLRenderTarget* target = nullptr;
};
struct ViewportCommand {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float minDepth = 0.0f;
    float maxDepth = 1.0f;
};
struct ScissorCommand {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};
struct LineWidthCommand { float width = 1.0f; };
struct SetPipelineCommand {
    OpenGLGraphicsPipeline* pipeline = nullptr;
};
struct SetBindGroupCommand {
    uint32_t index = 0;
    OpenGLBindGroup* group = nullptr;
};
struct SetVertexBufferCommand {
    uint32_t slot = 0;
    OpenGLBuffer* buffer = nullptr;
    uint64_t offset = 0;
};
struct SetIndexBufferCommand {
    OpenGLBuffer* buffer = nullptr;
    IndexFormat format = IndexFormat::Uint32;
    uint64_t offset = 0;
};
struct DrawCommand {
    uint32_t vertexCount = 0;
    uint32_t instanceCount = 1;
    uint32_t firstVertex = 0;
    PrimitiveTopology topology = PrimitiveTopology::TriangleList;
};
struct DrawIndexedCommand {
    uint32_t indexCount = 0;
    uint32_t instanceCount = 1;
    uint32_t firstIndex = 0;
    int32_t baseVertex = 0;
    IndexFormat format = IndexFormat::Uint32;
    uint64_t bufferOffset = 0;
    PrimitiveTopology topology = PrimitiveTopology::TriangleList;
};
struct EndPassCommand {
    OpenGLRenderTarget* target = nullptr;
};

using OpenGLCommand =
    std::variant<BeginPassCommand, ViewportCommand, ScissorCommand,
                 LineWidthCommand,
                 SetPipelineCommand, SetBindGroupCommand,
                 SetVertexBufferCommand, SetIndexBufferCommand, DrawCommand,
                 DrawIndexedCommand, EndPassCommand>;

struct OpenGLRecordingState {
    std::vector<OpenGLCommand> commands;
    bool passActive = false;
    OpenGLRenderTarget* activeTarget = nullptr;
    bool pipelineSet = false;
    OpenGLGraphicsPipeline* activePipeline = nullptr;
    std::vector<bool> vertexBuffersSet;
    std::vector<bool> bindGroupsSet;
    bool indexBufferSet = false;
    OpenGLBuffer* activeIndexBuffer = nullptr;
    IndexFormat indexFormat = IndexFormat::Uint32;
    uint64_t indexBufferOffset = 0;
    bool finished = false;
};

class OpenGLCommandBuffer final : public CommandBuffer {
  public:
    explicit OpenGLCommandBuffer(std::vector<OpenGLCommand> commands)
        : commands(std::move(commands)) {}

    std::vector<OpenGLCommand> commands;
    bool submitted = false;
};

class OpenGLRenderPassEncoder final : public RenderPassEncoder {
  public:
    explicit OpenGLRenderPassEncoder(
        std::shared_ptr<OpenGLRecordingState> state)
        : _state(std::move(state)) {}

    ~OpenGLRenderPassEncoder() override = default;

    void setViewport(float x, float y, float width, float height,
                     float minDepth, float maxDepth) override {
        requireActive();
        if (width < 0.0f || height < 0.0f || minDepth < 0.0f ||
            maxDepth > 1.0f || minDepth > maxDepth)
            throw std::invalid_argument("invalid render-pass viewport");
        _state->commands.emplace_back(
            ViewportCommand{x, y, width, height, minDepth, maxDepth});
    }

    void setScissor(uint32_t x, uint32_t y, uint32_t width,
                    uint32_t height) override {
        requireActive();
        _state->commands.emplace_back(ScissorCommand{x, y, width, height});
    }

    void setLineWidth(float width) override {
        requireActive();
        if (width < 1.0f)
            throw std::invalid_argument("line width must be at least 1");
        _state->commands.emplace_back(LineWidthCommand{width});
    }

    void setPipeline(GraphicsPipeline* pipeline) override {
        requireActive();
        auto* glPipeline = dynamic_cast<OpenGLGraphicsPipeline*>(pipeline);
        if (!glPipeline)
            throw std::invalid_argument("pass requires an OpenGL pipeline");
        const auto& pipelineDesc = glPipeline->getDesc();
        const auto& passDesc = _state->activeTarget->getDesc();
        if (pipelineDesc.sampleCount != _state->activeTarget->getSampleCount())
            throw std::invalid_argument("pipeline/pass sample count mismatch");
        if (pipelineDesc.colorTargets.size() !=
            passDesc.colorAttachments.size())
            throw std::invalid_argument(
                "pipeline/pass color target count mismatch");
        for (size_t index = 0; index < pipelineDesc.colorTargets.size();
             ++index) {
            const TextureFormat format = passDesc.colorAttachments[index]
                                             .view->getTexture()
                                             ->getFormat();
            if (pipelineDesc.colorTargets[index].format != format)
                throw std::invalid_argument(
                    "pipeline/pass color format mismatch");
        }
        if (pipelineDesc.depthStencil.has_value() !=
            passDesc.depthStencilAttachment.has_value())
            throw std::invalid_argument(
                "pipeline/pass depth attachment mismatch");
        if (pipelineDesc.depthStencil &&
            pipelineDesc.depthStencil->format !=
                passDesc.depthStencilAttachment->view->getTexture()
                    ->getFormat())
            throw std::invalid_argument("pipeline/pass depth format mismatch");
        _state->commands.emplace_back(SetPipelineCommand{glPipeline});
        _state->pipelineSet = true;
        _state->activePipeline = glPipeline;
        _state->vertexBuffersSet.assign(
            glPipeline->getDesc().vertexBuffers.size(), false);
        for (size_t slot = 0;
             slot < glPipeline->getDesc().vertexBuffers.size(); ++slot) {
            if (glPipeline->getDesc().vertexBuffers[slot].attributes.empty())
                _state->vertexBuffersSet[slot] = true;
        }
        const size_t groupCount = glPipeline->getDesc().pipelineLayout
                                      ? glPipeline->getDesc()
                                            .pipelineLayout->getDesc()
                                            .bindGroupLayouts.size()
                                      : 0;
        _state->bindGroupsSet.assign(groupCount, false);
        if (glPipeline->getDesc().pipelineLayout)
            for (size_t group = 0; group < groupCount; ++group)
                if (glPipeline->getDesc()
                        .pipelineLayout->getDesc()
                        .bindGroupLayouts[group]
                        ->getDesc()
                        .entries.empty())
                    _state->bindGroupsSet[group] = true;
        _state->indexBufferSet = false;
        _state->activeIndexBuffer = nullptr;
    }

    void setBindGroup(uint32_t index, BindGroup* bindGroup) override {
        requireActive();
        if (!_state->pipelineSet)
            throw std::logic_error("setBindGroup requires a pipeline");
        auto* glGroup = dynamic_cast<OpenGLBindGroup*>(bindGroup);
        if (!glGroup || index >= _state->bindGroupsSet.size() ||
            glGroup->getDesc().layout != _state->activePipeline->getDesc()
                                             .pipelineLayout->getDesc()
                                             .bindGroupLayouts[index])
            throw std::invalid_argument(
                "bind group is incompatible with pipeline");
        _state->commands.emplace_back(SetBindGroupCommand{index, glGroup});
        _state->bindGroupsSet[index] = true;
    }

    void setVertexBuffer(uint32_t slot, Buffer* buffer,
                         uint64_t offset) override {
        requireActive();
        if (!_state->pipelineSet)
            throw std::logic_error("setVertexBuffer requires a pipeline");
        auto* glBuffer = dynamic_cast<OpenGLBuffer*>(buffer);
        if (!glBuffer || !hasFlag(glBuffer->getUsage(), BufferUsage::Vertex))
            throw std::invalid_argument("buffer is missing Vertex usage");
        if (slot >= _state->vertexBuffersSet.size())
            throw std::invalid_argument("vertex buffer slot is not declared");
        if (offset >= glBuffer->getSize())
            throw std::invalid_argument(
                "vertex buffer offset is out of bounds");
        _state->commands.emplace_back(
            SetVertexBufferCommand{slot, glBuffer, offset});
        _state->vertexBuffersSet[slot] = true;
    }

    void setIndexBuffer(Buffer* buffer, IndexFormat format,
                        uint64_t offset) override {
        requireActive();
        if (!_state->pipelineSet)
            throw std::logic_error("setIndexBuffer requires a pipeline");
        auto* glBuffer = dynamic_cast<OpenGLBuffer*>(buffer);
        if (!glBuffer || !hasFlag(glBuffer->getUsage(), BufferUsage::Index))
            throw std::invalid_argument("buffer is missing Index usage");
        if (offset >= glBuffer->getSize())
            throw std::invalid_argument("index buffer offset is out of bounds");
        _state->commands.emplace_back(
            SetIndexBufferCommand{glBuffer, format, offset});
        _state->indexBufferSet = true;
        _state->activeIndexBuffer = glBuffer;
        _state->indexFormat = format;
        _state->indexBufferOffset = offset;
    }

    void draw(uint32_t vertexCount, uint32_t instanceCount,
              uint32_t firstVertex, uint32_t firstInstance) override {
        requireActive();
        if (!_state->pipelineSet)
            throw std::logic_error("draw requires a compatible pipeline");
        if (std::find(_state->vertexBuffersSet.begin(),
                      _state->vertexBuffersSet.end(),
                      false) != _state->vertexBuffersSet.end())
            throw std::logic_error(
                "draw requires every pipeline vertex buffer");
        if (std::find(_state->bindGroupsSet.begin(),
                      _state->bindGroupsSet.end(),
                      false) != _state->bindGroupsSet.end())
            throw std::logic_error("draw requires every pipeline bind group");
        if (vertexCount == 0 || instanceCount == 0)
            throw std::invalid_argument("draw counts must be non-zero");
        if (firstInstance != 0)
            throw std::invalid_argument(
                "non-zero firstInstance is unsupported");
        _state->commands.emplace_back(
            DrawCommand{vertexCount, instanceCount, firstVertex,
                        _state->activePipeline->getDesc().primitive.topology});
    }

    void drawIndexed(uint32_t indexCount, uint32_t instanceCount,
                     uint32_t firstIndex, int32_t baseVertex,
                     uint32_t firstInstance) override {
        requireActive();
        if (!_state->pipelineSet)
            throw std::logic_error("drawIndexed requires a pipeline");
        if (!_state->indexBufferSet)
            throw std::logic_error("drawIndexed requires an index buffer");
        if (std::find(_state->vertexBuffersSet.begin(),
                      _state->vertexBuffersSet.end(),
                      false) != _state->vertexBuffersSet.end())
            throw std::logic_error(
                "drawIndexed requires every pipeline vertex buffer");
        if (std::find(_state->bindGroupsSet.begin(),
                      _state->bindGroupsSet.end(),
                      false) != _state->bindGroupsSet.end())
            throw std::logic_error(
                "drawIndexed requires every pipeline bind group");
        if (indexCount == 0 || instanceCount == 0)
            throw std::invalid_argument("drawIndexed counts must be non-zero");
        if (firstInstance != 0)
            throw std::invalid_argument(
                "non-zero firstInstance is unsupported");
        const uint64_t elementSize =
            _state->indexFormat == IndexFormat::Uint16 ? 2u : 4u;
        const uint64_t required =
            _state->indexBufferOffset +
            (static_cast<uint64_t>(firstIndex) + indexCount) * elementSize;
        if (required > _state->activeIndexBuffer->getSize())
            throw std::invalid_argument(
                "indexed draw exceeds index buffer size");
        _state->commands.emplace_back(DrawIndexedCommand{
            indexCount, instanceCount, firstIndex, baseVertex,
            _state->indexFormat, _state->indexBufferOffset,
            _state->activePipeline->getDesc().primitive.topology});
    }

    void end() override {
        requireActive();
        _state->commands.emplace_back(
            EndPassCommand{_state->activeTarget});
        _state->passActive = false;
        _state->activeTarget = nullptr;
        _state->pipelineSet = false;
        _state->activePipeline = nullptr;
        _state->vertexBuffersSet.clear();
        _state->bindGroupsSet.clear();
        _state->indexBufferSet = false;
        _state->activeIndexBuffer = nullptr;
        _ended = true;
    }

  private:
    void requireActive() const {
        if (_ended || !_state || _state->finished || !_state->passActive)
            throw std::logic_error("render pass encoder is not active");
    }

    std::shared_ptr<OpenGLRecordingState> _state;
    bool _ended = false;
};

class OpenGLCommandEncoder final : public CommandEncoder {
  public:
    OpenGLCommandEncoder() : _state(std::make_shared<OpenGLRecordingState>()) {}

    std::unique_ptr<RenderPassEncoder>
    beginRenderPass(RenderTarget* target) override {
        requireRecording();
        if (_state->passActive)
            throw std::logic_error("nested render passes are not allowed");
        auto* glTarget = dynamic_cast<OpenGLRenderTarget*>(target);
        if (!glTarget)
            throw std::invalid_argument(
                "OpenGL command encoder requires an OpenGL render target");
        _state->commands.emplace_back(BeginPassCommand{glTarget});
        _state->passActive = true;
        _state->activeTarget = glTarget;
        _state->pipelineSet = false;
        _state->activePipeline = nullptr;
        _state->vertexBuffersSet.clear();
        _state->bindGroupsSet.clear();
        _state->indexBufferSet = false;
        _state->activeIndexBuffer = nullptr;
        return std::make_unique<OpenGLRenderPassEncoder>(_state);
    }

    std::unique_ptr<CommandBuffer> finish() override {
        requireRecording();
        if (_state->passActive)
            throw std::logic_error(
                "cannot finish a command encoder with an active render pass");
        _state->finished = true;
        return std::make_unique<OpenGLCommandBuffer>(
            std::move(_state->commands));
    }

  private:
    void requireRecording() const {
        if (!_state || _state->finished)
            throw std::logic_error("command encoder is already finished");
    }

    std::shared_ptr<OpenGLRecordingState> _state;
};

} // namespace

// OpenGLDevice Implementation
OpenGLDevice::OpenGLDevice() : _initialized(false) {}

OpenGLDevice::~OpenGLDevice() {
    if (_skyboxVAO)
        glDeleteVertexArrays(1, &_skyboxVAO);
    if (_skyboxTex)
        glDeleteTextures(1, &_skyboxTex);
    if (_initialized)
        shutdown();
}

void OpenGLDevice::initialize() {
    if (_initialized)
        return;

    // OpenGL context should already be created by Window
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    // TODO: need material-wise different culling
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    _renderThread = std::this_thread::get_id();
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

void OpenGLDevice::drawLines(size_t vertexCount) {
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertexCount));
}

void OpenGLDevice::drawPoints(size_t vertexCount) {
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertexCount));
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

std::unique_ptr<Buffer> OpenGLDevice::createBuffer(const BufferDesc& desc,
                                                   const void* data) {
    if (!_initialized || std::this_thread::get_id() != _renderThread)
        throw std::runtime_error(
            "OpenGL buffers must be created on the render thread");
    return std::make_unique<OpenGLBuffer>(desc, data);
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

std::unique_ptr<Texture>
OpenGLDevice::createTexture(const TextureDesc& desc,
                            const SamplerDesc& sampler) {
    return std::make_unique<OpenGLTexture>(desc, sampler);
}

std::unique_ptr<Texture>
OpenGLDevice::createTexture(const TextureResourceDesc& desc,
                            const TextureInitialData* initialData) {
    return std::make_unique<OpenGLTexture>(desc, initialData);
}

std::unique_ptr<TextureView>
OpenGLDevice::createTextureView(Texture* texture, const TextureViewDesc& desc) {
    auto* glTexture = dynamic_cast<OpenGLTexture*>(texture);
    if (!glTexture)
        throw std::invalid_argument(
            "OpenGLDevice::createTextureView requires an OpenGL texture");
    return std::make_unique<OpenGLTextureView>(glTexture, desc);
}

std::unique_ptr<Sampler> OpenGLDevice::createSampler(const SamplerDesc& desc) {
    return std::make_unique<OpenGLSampler>(desc);
}

std::unique_ptr<RenderTarget>
OpenGLDevice::createRenderTarget(const RenderPassDesc& desc) {
    return std::make_unique<OpenGLRenderTarget>(desc);
}

void OpenGLDevice::beginLegacyRenderPass(RenderTarget* target) {
    if (std::this_thread::get_id() != _renderThread)
        throw std::runtime_error(
            "legacy RHI render pass must begin on the render thread");
    auto* glTarget = dynamic_cast<OpenGLRenderTarget*>(target);
    if (!glTarget)
        throw std::invalid_argument(
            "legacy RHI render pass requires an OpenGL target");
    glTarget->beginPass();
}

void OpenGLDevice::endLegacyRenderPass(RenderTarget* target) {
    if (std::this_thread::get_id() != _renderThread)
        throw std::runtime_error(
            "legacy RHI render pass must end on the render thread");
    auto* glTarget = dynamic_cast<OpenGLRenderTarget*>(target);
    if (!glTarget)
        throw std::invalid_argument(
            "legacy RHI render pass requires an OpenGL target");
    glTarget->endPass();
}

std::unique_ptr<GraphicsPipeline>
OpenGLDevice::createGraphicsPipeline(const GraphicsPipelineDesc& desc) {
    if (!_initialized || std::this_thread::get_id() != _renderThread)
        throw std::runtime_error(
            "OpenGL pipelines must be created on the render thread");
    return std::make_unique<OpenGLGraphicsPipeline>(desc);
}

std::unique_ptr<BindGroupLayout>
OpenGLDevice::createBindGroupLayout(const BindGroupLayoutDesc& desc) {
    return std::make_unique<OpenGLBindGroupLayout>(desc);
}

std::unique_ptr<PipelineLayout>
OpenGLDevice::createPipelineLayout(const PipelineLayoutDesc& desc) {
    return std::make_unique<OpenGLPipelineLayout>(desc);
}

std::unique_ptr<BindGroup>
OpenGLDevice::createBindGroup(const BindGroupDesc& desc) {
    return std::make_unique<OpenGLBindGroup>(desc);
}

TextureReadback OpenGLDevice::readTexture(TextureView* view) {
    if (!_initialized || std::this_thread::get_id() != _renderThread)
        throw std::runtime_error(
            "OpenGL texture readback must run on the render thread");
    auto* glView = dynamic_cast<OpenGLTextureView*>(view);
    if (!glView)
        throw std::invalid_argument(
            "readTexture requires an OpenGL texture view");
    auto* texture = dynamic_cast<OpenGLTexture*>(glView->getTexture());
    if (!hasFlag(texture->getUsage(), TextureUsage::CopySrc))
        throw std::invalid_argument(
            "readTexture requires CopySrc texture usage");

    TextureReadback result;
    result.width = static_cast<uint32_t>(texture->getWidth());
    result.height = static_cast<uint32_t>(texture->getHeight());
    result.format = texture->getFormat();
    result.aspect = glView->getDesc().aspect;
    const bool depth = isDepthFormat(result.format);
    result.componentCount =
        depth || result.format == TextureFormat::R8Unorm ? 1u : 4u;
    result.values.resize(static_cast<size_t>(result.width) * result.height *
                         result.componentCount);

    GLint previousDrawFbo = 0;
    GLint previousReadFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFbo);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousReadFbo);
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    if (depth) {
        const GLenum attachment = hasStencilAspect(result.format)
                                      ? GL_DEPTH_STENCIL_ATTACHMENT
                                      : GL_DEPTH_ATTACHMENT;
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, attachment, GL_TEXTURE_2D,
                               texture->getHandle(), 0);
        glReadBuffer(GL_NONE);
    } else {
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, texture->getHandle(), 0);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
    }
    if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) !=
        GL_FRAMEBUFFER_COMPLETE) {
        glDeleteFramebuffers(1, &fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previousDrawFbo);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, previousReadFbo);
        throw std::runtime_error("readTexture framebuffer is incomplete");
    }
    glReadPixels(0, 0, result.width, result.height,
                 depth ? GL_DEPTH_COMPONENT
                       : (result.componentCount == 1 ? GL_RED : GL_RGBA),
                 GL_FLOAT, result.values.data());
    glDeleteFramebuffers(1, &fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previousDrawFbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, previousReadFbo);
    return result;
}

std::unique_ptr<CommandEncoder> OpenGLDevice::createCommandEncoder() {
    return std::make_unique<OpenGLCommandEncoder>();
}

void OpenGLDevice::submit(CommandBuffer& commandBuffer) {
    if (!_initialized || std::this_thread::get_id() != _renderThread)
        throw std::runtime_error(
            "OpenGL command buffers must be submitted on the render thread");
    auto* glCommands = dynamic_cast<OpenGLCommandBuffer*>(&commandBuffer);
    if (!glCommands)
        throw std::invalid_argument(
            "OpenGLDevice::submit requires an OpenGL command buffer");
    if (glCommands->submitted)
        throw std::logic_error("command buffer has already been submitted");
    if (_validationEnabled)
        while (glGetError() != GL_NO_ERROR) {
        }

    GLint previousDrawFbo = 0;
    GLint previousReadFbo = 0;
    GLint previousViewport[4] = {};
    GLint previousScissor[4] = {};
    GLdouble previousDepthRange[2] = {};
    const GLboolean previousScissorEnabled = glIsEnabled(GL_SCISSOR_TEST);
    const GLboolean previousDepthEnabled = glIsEnabled(GL_DEPTH_TEST);
    const GLboolean previousCullEnabled = glIsEnabled(GL_CULL_FACE);
    const GLboolean previousBlendEnabled = glIsEnabled(GL_BLEND);
    GLboolean previousDepthMask = GL_TRUE;
    GLboolean previousColorMask[4] = {};
    GLint previousDepthFunc = GL_LESS;
    GLint previousCullFace = GL_BACK;
    GLint previousFrontFace = GL_CCW;
    GLfloat previousLineWidth = 1.0f;
    GLint previousBlendSrcRgb = GL_ONE;
    GLint previousBlendDstRgb = GL_ZERO;
    GLint previousBlendSrcAlpha = GL_ONE;
    GLint previousBlendDstAlpha = GL_ZERO;
    GLint previousBlendEquationRgb = GL_FUNC_ADD;
    GLint previousBlendEquationAlpha = GL_FUNC_ADD;
    GLint previousPolygonMode[2] = {};
    GLint previousProgram = 0;
    GLint previousVao = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousDrawFbo);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousReadFbo);
    glGetIntegerv(GL_VIEWPORT, previousViewport);
    glGetIntegerv(GL_SCISSOR_BOX, previousScissor);
    glGetDoublev(GL_DEPTH_RANGE, previousDepthRange);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &previousDepthMask);
    glGetBooleanv(GL_COLOR_WRITEMASK, previousColorMask);
    glGetIntegerv(GL_DEPTH_FUNC, &previousDepthFunc);
    glGetIntegerv(GL_CULL_FACE_MODE, &previousCullFace);
    glGetIntegerv(GL_FRONT_FACE, &previousFrontFace);
    glGetFloatv(GL_LINE_WIDTH, &previousLineWidth);
    glGetIntegerv(GL_BLEND_SRC_RGB, &previousBlendSrcRgb);
    glGetIntegerv(GL_BLEND_DST_RGB, &previousBlendDstRgb);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &previousBlendSrcAlpha);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &previousBlendDstAlpha);
    glGetIntegerv(GL_BLEND_EQUATION_RGB, &previousBlendEquationRgb);
    glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &previousBlendEquationAlpha);
    glGetIntegerv(GL_POLYGON_MODE, previousPolygonMode);
    glGetIntegerv(GL_CURRENT_PROGRAM, &previousProgram);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &previousVao);

    GLint previousActiveTexture = GL_TEXTURE0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &previousActiveTexture);
    std::vector<GLuint> usedTextureSlots;
    std::vector<GLuint> usedUniformSlots;
    for (const OpenGLCommand& command : glCommands->commands) {
        if (const auto* set = std::get_if<SetPipelineCommand>(&command)) {
            for (GLuint slot : set->pipeline->textureSlots())
                if (std::find(usedTextureSlots.begin(), usedTextureSlots.end(),
                              slot) == usedTextureSlots.end())
                    usedTextureSlots.push_back(slot);
            for (GLuint slot : set->pipeline->uniformSlots())
                if (std::find(usedUniformSlots.begin(), usedUniformSlots.end(),
                              slot) == usedUniformSlots.end())
                    usedUniformSlots.push_back(slot);
        }
    }
    struct TextureBindingState {
        GLuint texture2D = 0;
        GLuint textureCube = 0;
        GLuint sampler = 0;
    };
    std::vector<TextureBindingState> textureBindingStates;
    for (GLuint slot : usedTextureSlots) {
        glActiveTexture(GL_TEXTURE0 + slot);
        GLint texture2D = 0;
        GLint textureCube = 0;
        GLint sampler = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &texture2D);
        glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &textureCube);
        glGetIntegeri_v(GL_SAMPLER_BINDING, slot, &sampler);
        textureBindingStates.push_back(
            {static_cast<GLuint>(texture2D), static_cast<GLuint>(textureCube),
             static_cast<GLuint>(sampler)});
    }
    std::vector<GLuint> uniformBindingStates;
    for (GLuint slot : usedUniformSlots) {
        GLint buffer = 0;
        glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, slot, &buffer);
        uniformBindingStates.push_back(static_cast<GLuint>(buffer));
    }
    glActiveTexture(previousActiveTexture);
    auto restoreBindings = [&] {
        for (size_t index = 0; index < usedTextureSlots.size(); ++index) {
            glActiveTexture(GL_TEXTURE0 + usedTextureSlots[index]);
            glBindTexture(GL_TEXTURE_2D,
                          textureBindingStates[index].texture2D);
            glBindTexture(GL_TEXTURE_CUBE_MAP,
                          textureBindingStates[index].textureCube);
            glBindSampler(usedTextureSlots[index],
                          textureBindingStates[index].sampler);
        }
        for (size_t index = 0; index < usedUniformSlots.size(); ++index)
            glBindBufferBase(GL_UNIFORM_BUFFER, usedUniformSlots[index],
                             uniformBindingStates[index]);
        glActiveTexture(previousActiveTexture);
    };

    bool passActive = false;
    OpenGLGraphicsPipeline* activePipeline = nullptr;
    try {
        for (const OpenGLCommand& command : glCommands->commands) {
            std::visit(
                [&](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, BeginPassCommand>) {
                        value.target->beginPass();
                        passActive = true;
                        if (glPushDebugGroup != nullptr) {
                            const std::string& label =
                                value.target->getDesc().label;
                            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1,
                                             label.empty() ? "RenderPass"
                                                           : label.c_str());
                        }
                    } else if constexpr (std::is_same_v<T, ViewportCommand>) {
                        glViewport(static_cast<GLint>(value.x),
                                   static_cast<GLint>(value.y),
                                   static_cast<GLsizei>(value.width),
                                   static_cast<GLsizei>(value.height));
                        glDepthRange(value.minDepth, value.maxDepth);
                    } else if constexpr (std::is_same_v<T, ScissorCommand>) {
                        glEnable(GL_SCISSOR_TEST);
                        glScissor(static_cast<GLint>(value.x),
                                  static_cast<GLint>(value.y),
                                  static_cast<GLsizei>(value.width),
                                  static_cast<GLsizei>(value.height));
                    } else if constexpr (std::is_same_v<T,
                                                        LineWidthCommand>) {
                        glLineWidth(value.width);
                    } else if constexpr (std::is_same_v<T,
                                                        SetPipelineCommand>) {
                        value.pipeline->apply();
                        activePipeline = value.pipeline;
                    } else if constexpr (std::is_same_v<T,
                                                        SetBindGroupCommand>) {
                        activePipeline->bindGroup(value.index, value.group);
                    } else if constexpr (std::is_same_v<
                                             T, SetVertexBufferCommand>) {
                        activePipeline->bindVertexBuffer(
                            value.slot, value.buffer, value.offset);
                    } else if constexpr (std::is_same_v<
                                             T, SetIndexBufferCommand>) {
                        activePipeline->bindIndexBuffer(value.buffer);
                    } else if constexpr (std::is_same_v<T, DrawCommand>) {
                        glDrawArraysInstanced(
                            toGLTopology(value.topology),
                            static_cast<GLint>(value.firstVertex),
                            static_cast<GLsizei>(value.vertexCount),
                            static_cast<GLsizei>(value.instanceCount));
                    } else if constexpr (std::is_same_v<T,
                                                        DrawIndexedCommand>) {
                        const GLenum type = value.format == IndexFormat::Uint16
                                                ? GL_UNSIGNED_SHORT
                                                : GL_UNSIGNED_INT;
                        const uint64_t elementSize =
                            value.format == IndexFormat::Uint16 ? 2u : 4u;
                        const auto* indices = reinterpret_cast<const void*>(
                            value.bufferOffset +
                            static_cast<uint64_t>(value.firstIndex) *
                                elementSize);
                        glDrawElementsInstancedBaseVertex(
                            toGLTopology(value.topology),
                            static_cast<GLsizei>(value.indexCount), type,
                            indices, static_cast<GLsizei>(value.instanceCount),
                            value.baseVertex);
                    } else if constexpr (std::is_same_v<T, EndPassCommand>) {
                        if (glPopDebugGroup != nullptr)
                            glPopDebugGroup();
                        value.target->endPass();
                        passActive = false;
                        activePipeline = nullptr;
                    }
                },
                command);
        }
        glCommands->submitted = true;
    } catch (...) {
        if (passActive && glPopDebugGroup != nullptr)
            glPopDebugGroup();
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previousDrawFbo);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, previousReadFbo);
        glViewport(previousViewport[0], previousViewport[1],
                   previousViewport[2], previousViewport[3]);
        glDepthRange(previousDepthRange[0], previousDepthRange[1]);
        previousScissorEnabled ? glEnable(GL_SCISSOR_TEST)
                               : glDisable(GL_SCISSOR_TEST);
        previousDepthEnabled ? glEnable(GL_DEPTH_TEST)
                             : glDisable(GL_DEPTH_TEST);
        previousCullEnabled ? glEnable(GL_CULL_FACE) : glDisable(GL_CULL_FACE);
        previousBlendEnabled ? glEnable(GL_BLEND) : glDisable(GL_BLEND);
        glBlendEquationSeparate(previousBlendEquationRgb,
                                previousBlendEquationAlpha);
        glBlendFuncSeparate(previousBlendSrcRgb, previousBlendDstRgb,
                            previousBlendSrcAlpha, previousBlendDstAlpha);
        glDepthMask(previousDepthMask);
        glColorMask(previousColorMask[0], previousColorMask[1],
                    previousColorMask[2], previousColorMask[3]);
        glDepthFunc(previousDepthFunc);
        glCullFace(previousCullFace);
        glFrontFace(previousFrontFace);
        glLineWidth(previousLineWidth);
        glPolygonMode(GL_FRONT_AND_BACK, previousPolygonMode[0]);
        glUseProgram(previousProgram);
        glBindVertexArray(previousVao);
        restoreBindings();
        glScissor(previousScissor[0], previousScissor[1], previousScissor[2],
                  previousScissor[3]);
        throw;
    }

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previousDrawFbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, previousReadFbo);
    glViewport(previousViewport[0], previousViewport[1], previousViewport[2],
               previousViewport[3]);
    glDepthRange(previousDepthRange[0], previousDepthRange[1]);
    previousScissorEnabled ? glEnable(GL_SCISSOR_TEST)
                           : glDisable(GL_SCISSOR_TEST);
    previousDepthEnabled ? glEnable(GL_DEPTH_TEST) : glDisable(GL_DEPTH_TEST);
    previousCullEnabled ? glEnable(GL_CULL_FACE) : glDisable(GL_CULL_FACE);
    previousBlendEnabled ? glEnable(GL_BLEND) : glDisable(GL_BLEND);
    glBlendEquationSeparate(previousBlendEquationRgb,
                            previousBlendEquationAlpha);
    glBlendFuncSeparate(previousBlendSrcRgb, previousBlendDstRgb,
                        previousBlendSrcAlpha, previousBlendDstAlpha);
    glDepthMask(previousDepthMask);
    glColorMask(previousColorMask[0], previousColorMask[1],
                previousColorMask[2], previousColorMask[3]);
    glDepthFunc(previousDepthFunc);
    glCullFace(previousCullFace);
    glFrontFace(previousFrontFace);
    glLineWidth(previousLineWidth);
    glPolygonMode(GL_FRONT_AND_BACK, previousPolygonMode[0]);
    glUseProgram(previousProgram);
    glBindVertexArray(previousVao);
    restoreBindings();
    glScissor(previousScissor[0], previousScissor[1], previousScissor[2],
              previousScissor[3]);
    if (_validationEnabled) {
        const GLenum error = glGetError();
        if (error != GL_NO_ERROR)
            throw std::runtime_error(
                "OpenGL validation error after command submission: " +
                std::to_string(static_cast<uint32_t>(error)));
    }
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

std::unique_ptr<Texture>
OpenGLDevice::createTexture(const std::string path, bool flip,
                            const SamplerDesc& sampler) {
    TextureDesc desc = loadImage(path, flip);
    auto texture = std::make_unique<OpenGLTexture>(desc, sampler);
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

void OpenGLDevice::setColorWrite(bool enable) {
    glColorMask(enable, enable, enable, enable);
}

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

void OpenGLDevice::setStencilFunc(StencilFunc func, int ref, uint32_t mask) {
    auto toGL = [](StencilFunc f) -> GLenum {
        switch (f) {
        case StencilFunc::Never:
            return GL_NEVER;
        case StencilFunc::Less:
            return GL_LESS;
        case StencilFunc::LessEqual:
            return GL_LEQUAL;
        case StencilFunc::Greater:
            return GL_GREATER;
        case StencilFunc::GreaterEqual:
            return GL_GEQUAL;
        case StencilFunc::Equal:
            return GL_EQUAL;
        case StencilFunc::NotEqual:
            return GL_NOTEQUAL;
        case StencilFunc::Always:
            return GL_ALWAYS;
        }
        return GL_ALWAYS;
    };
    glStencilFunc(toGL(func), ref, mask);
}

void OpenGLDevice::setStencilOp(StencilOp stencilFail, StencilOp depthFail,
                                StencilOp depthPass) {
    auto toGL = [](StencilOp op) -> GLenum {
        switch (op) {
        case StencilOp::Keep:
            return GL_KEEP;
        case StencilOp::Zero:
            return GL_ZERO;
        case StencilOp::Replace:
            return GL_REPLACE;
        case StencilOp::IncrementClamp:
            return GL_INCR;
        case StencilOp::DecrementClamp:
            return GL_DECR;
        case StencilOp::Invert:
            return GL_INVERT;
        case StencilOp::IncrementWrap:
            return GL_INCR_WRAP;
        case StencilOp::DecrementWrap:
            return GL_DECR_WRAP;
        }
        return GL_KEEP;
    };
    glStencilOp(toGL(stencilFail), toGL(depthFail), toGL(depthPass));
}

void OpenGLDevice::setStencilWriteMask(uint32_t mask) { glStencilMask(mask); }

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

void OpenGLDevice::setLineWidth(float width) {
    GLfloat range[2] = {1.0f, 1.0f};
    glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, range);
    glLineWidth(std::clamp(width, range[0], range[1]));
}

void OpenGLDevice::setCullFace(bool enable) {
    if (enable)
        glEnable(GL_CULL_FACE);
    else
        glDisable(GL_CULL_FACE);
}

void OpenGLDevice::setCullFaceMode(CullFaceMode mode) {
    GLenum glMode;
    switch (mode) {
    case CullFaceMode::Front:
        glMode = GL_FRONT;
        break;
    case CullFaceMode::Back:
        glMode = GL_BACK;
        break;
    }
    glCullFace(glMode);
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

    if (attribute.type == VertexAttributeType::Int ||
        attribute.type == VertexAttributeType::UnsignedInt) {
        glVertexAttribIPointer(attribute.location, attribute.size, glType,
                               attribute.stride,
                               reinterpret_cast<void*>(attribute.offset));
    } else {
        glVertexAttribPointer(attribute.location, attribute.size, glType,
                              attribute.normalized ? GL_TRUE : GL_FALSE,
                              attribute.stride,
                              reinterpret_cast<void*>(attribute.offset));
    }
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

    if (!_desc.depthOnly) {
        _colorTexObj = std::make_unique<OpenGLTexture>(
            _desc.width, _desc.height, _desc.colorFormat);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, _colorTexObj->getHandle(), 0);
    } else {
        // depth-only FBO (shadow map): no color writes or reads
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
    }

    // depth (or depth+stencil) texture — required for shadow map sampling
    GLenum depthAtt =
        _desc.stencil ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT;
    _depthTexObj = std::make_unique<OpenGLTexture>(_desc.width, _desc.height,
                                                   _desc.stencil);
    glFramebufferTexture2D(GL_FRAMEBUFFER, depthAtt, GL_TEXTURE_2D,
                           _depthTexObj->getHandle(), 0);

    // [SIMPLE RBO] depth+stencil renderbuffer (faster, but no shader
    // sampling) GLuint rbo; glGenRenderbuffers(1, &rbo);
    // glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    // glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8,
    //                       _desc.width, _desc.height);
    // glFramebufferRenderbuffer(GL_FRAMEBUFFER,
    // GL_DEPTH_STENCIL_ATTACHMENT,
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
        const GLFramebufferColorFormat glColorFormat =
            toGLFramebufferColorFormat(_desc.colorFormat);
        glGenRenderbuffers(1, &_msaaColorRbo);
        glBindRenderbuffer(GL_RENDERBUFFER, _msaaColorRbo);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, _desc.msaaSamples,
                                         glColorFormat.internalFormat,
                                         _desc.width, _desc.height);

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
    if (_fbo)
        glDeleteFramebuffers(1, &_fbo);
}

void OpenGLFramebuffer::bind() {
    // Render into MSAA FBO if available, otherwise directly into texture
    // FBO
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
    if (!_desc.depthOnly) {
        const GLFramebufferColorFormat glColorFormat =
            toGLFramebufferColorFormat(_desc.colorFormat);
        glBindTexture(GL_TEXTURE_2D, _colorTexObj->getHandle());
        glTexImage2D(GL_TEXTURE_2D, 0, glColorFormat.internalFormat, w, h, 0,
                     glColorFormat.format, glColorFormat.type, nullptr);
        _colorTexObj->setSize(w, h);
    }
    glBindTexture(GL_TEXTURE_2D, _depthTexObj->getHandle());
    {
        GLenum depthFmt =
            _desc.stencil ? GL_DEPTH24_STENCIL8 : GL_DEPTH_COMPONENT32;
        GLenum depthBase =
            _desc.stencil ? GL_DEPTH_STENCIL : GL_DEPTH_COMPONENT;
        GLenum depthType = _desc.stencil ? GL_UNSIGNED_INT_24_8 : GL_FLOAT;
        glTexImage2D(GL_TEXTURE_2D, 0, depthFmt, w, h, 0, depthBase, depthType,
                     nullptr);
        _depthTexObj->setSize(w, h);
    }

    // Resize MSAA RBOs
    if (_desc.msaaSamples > 0) {
        const GLFramebufferColorFormat glColorFormat =
            toGLFramebufferColorFormat(_desc.colorFormat);
        glBindRenderbuffer(GL_RENDERBUFFER, _msaaColorRbo);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, _desc.msaaSamples,
                                         glColorFormat.internalFormat, w, h);
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

std::vector<uint8_t> OpenGLFramebuffer::readColorPixels(bool flipY) {
    std::vector<uint8_t> pixels(static_cast<std::size_t>(_desc.width) *
                                static_cast<std::size_t>(_desc.height) * 3);
    if (_desc.depthOnly || _desc.width <= 0 || _desc.height <= 0)
        return pixels;

    std::vector<uint8_t> raw(pixels.size());
    glBindFramebuffer(GL_READ_FRAMEBUFFER, _fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    // TODO: more efficient color formats control for all system.
    glReadPixels(0, 0, _desc.width, _desc.height, GL_RGB, GL_UNSIGNED_BYTE,
                 raw.data());
    // restore GL state
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);

    for (int y = 0; y < _desc.height; ++y) {
        const int srcY = flipY ? (_desc.height - 1 - y) : y;
        const auto src = static_cast<std::size_t>(srcY) * _desc.width * 3;
        const auto dst = static_cast<std::size_t>(y) * _desc.width * 3;
        std::copy(raw.begin() + src, raw.begin() + src + _desc.width * 3,
                  pixels.begin() + dst);
    }
    return pixels;
}

// TODO: Add ping-pong PBO readback
std::vector<uint8_t>
OpenGLFramebuffer::readColorPixelsResized(int width, int height, bool flipY) {
    if (width <= 0 || height <= 0)
        throw std::invalid_argument(
            "readback width and height must be positive");
    if (width == _desc.width && height == _desc.height)
        return readColorPixels(flipY);

    if (!_scaledReadbackFramebuffer) {
        FramebufferDesc desc;
        desc.width = width;
        desc.height = height;
        desc.colorFormat = FramebufferColorFormat::RGBA8;
        _scaledReadbackFramebuffer = std::make_unique<OpenGLFramebuffer>(desc);
    } else {
        Texture* color = _scaledReadbackFramebuffer->getColorTexture();
        if (!color || color->getWidth() != width ||
            color->getHeight() != height)
            _scaledReadbackFramebuffer->resize(width, height);
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, _fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _scaledReadbackFramebuffer->_fbo);
    glBlitFramebuffer(0, 0, _desc.width, _desc.height, 0, 0, width, height,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return _scaledReadbackFramebuffer->readColorPixels(flipY);
}

Texture* OpenGLFramebuffer::getColorTexture() { return _colorTexObj.get(); }
Texture* OpenGLFramebuffer::getDepthTexture() { return _depthTexObj.get(); }
Texture* OpenGLFramebuffer::getStencilTexture() { return nullptr; }
Texture* OpenGLFramebuffer::getDepthStencilTexture() { return nullptr; }

// ---------------------------------------------------------------------------
// Skybox

static const char* skyboxVs = R"(
#version 410 core
layout(location = 0) in vec3 aPos;
out vec3 TexDir;
uniform mat4 projection;
uniform mat4 view;
uniform bool zUp;
void main() {
    // Y-up world: cube coords map directly to cubemap coords.
    // Z-up world: remap so that world +Z -> cubemap +Y (up),
    //             world -Y -> cubemap +Z (south), world +X stays.
    TexDir = zUp ? vec3(aPos.x, aPos.z, -aPos.y) : aPos;
    vec4 pos = projection * mat4(mat3(view)) * vec4(aPos, 1.0); // remove trans of view
    gl_Position = pos.xyww; // w / w = 1.0 early depth testing
}
)";

static const char* skyboxFs = R"(
#version 410 core
in vec3 TexDir;
out vec4 FragColor;
uniform samplerCube skybox;
void main() {
    FragColor = texture(skybox, TexDir);
}
)";

// Load a cross-layout(putting all images on one png) cubemap PNG and upload
// it as GL_TEXTURE_CUBE_MAP. Supports both horizontal cross (4:3, W>H) and
// vertical cross (3:4, H>W).
//
// Horizontal cross face layout (faceSize = W/4 = H/3):
//   col: 0=-X  1=+Y  2=+X  3=-Y   (row 0 only: col1)
//   row: 0=+Y  1=(-X,+Z,+X,-Z)   2=-Y
//   Actually standard H-cross:
//     row0,col1 = +Y
//     row1,col0 = -X   col1 = +Z   col2 = +X   col3 = -Z
//     row2,col1 = -Y
//
// Vertical cross face layout (faceSize = W/3 = H/4):
//     row0,col1 = +Y
//     row1,col0 = -X   col1 = +Z   col2 = +X
//     row2,col1 = -Y
//     row3,col1 = -Z
GLuint OpenGLDevice::loadCubemapCross(const std::string& path) {
    stbi_set_flip_vertically_on_load(false);
    int w, h, ch;
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &ch, 4);
    if (!img) {
        fprintf(stderr, "Cubemap load failed: %s\n", path.c_str());
        return 0;
    }
    ch = 4;

    int faceW, faceH;
    bool horizontal;
    if (w * 3 == h * 4) { // 4:3 -> horizontal cross
        faceW = w / 4;
        faceH = h / 3;
        horizontal = true;
    } else if (w * 4 == h * 3) { // 3:4 -> vertical cross
        faceW = w / 3;
        faceH = h / 4;
        horizontal = false;
    } else {
        // Fallback: assume horizontal cross, truncate
        faceW = w / 4;
        faceH = h / 3;
        horizontal = true;
        fprintf(stderr,
                "Cubemap: unexpected aspect %dx%d, assuming horizontal cross\n",
                w, h);
    }

    // face[i] = {col, row} in the cross grid
    // GL order: +X,-X,+Y,-Y,+Z,-Z
    int faceCol[6], faceRow[6];
    if (horizontal) {
        faceCol[0] = 2;
        faceRow[0] = 1; // +X
        faceCol[1] = 0;
        faceRow[1] = 1; // -X
        faceCol[2] = 1;
        faceRow[2] = 0; // +Y
        faceCol[3] = 1;
        faceRow[3] = 2; // -Y
        faceCol[4] = 1;
        faceRow[4] = 1; // +Z
        faceCol[5] = 3;
        faceRow[5] = 1; // -Z
    } else {
        faceCol[0] = 2;
        faceRow[0] = 1; // +X
        faceCol[1] = 0;
        faceRow[1] = 1; // -X
        faceCol[2] = 1;
        faceRow[2] = 0; // +Y
        faceCol[3] = 1;
        faceRow[3] = 2; // -Y
        faceCol[4] = 1;
        faceRow[4] = 1; // +Z
        faceCol[5] = 1;
        faceRow[5] = 3; // -Z
    }

    // Extract one face into a temporary buffer
    std::vector<unsigned char> face(faceW * faceH * ch);

    GLuint texID;
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texID);

    for (int f = 0; f < 6; f++) {
        int ox = faceCol[f] * faceW;
        int oy = faceRow[f] * faceH;
        for (int row = 0; row < faceH; row++) {
            memcpy(face.data() + row * faceW * ch,
                   img + ((oy + row) * w + ox) * ch, faceW * ch);
        }
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, 0, GL_RGBA, faceW,
                     faceH, 0, GL_RGBA, GL_UNSIGNED_BYTE, face.data());
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    stbi_image_free(img);
    return texID;
}

// Load 6 individual face images and upload as GL_TEXTURE_CUBE_MAP.
// paths order: +X, -X, +Y, -Y, +Z, -Z (matches
// GL_TEXTURE_CUBE_MAP_POSITIVE_X + i)
GLuint OpenGLDevice::loadCubemap(const std::vector<std::string>& paths) {
    if (paths.size() != 6) {
        fprintf(stderr, "loadCubemap: expected 6 paths, got %zu\n",
                paths.size());
        return 0;
    }
    GLuint texID;
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texID);

    stbi_set_flip_vertically_on_load(false);
    for (int f = 0; f < 6; f++) {
        int w, h, ch;
        unsigned char* img = stbi_load(paths[f].c_str(), &w, &h, &ch, 4);
        if (!img) {
            fprintf(stderr, "loadCubemap: failed to load face %d: %s\n", f,
                    paths[f].c_str());
            glDeleteTextures(1, &texID);
            return 0;
        }
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, img);
        stbi_image_free(img);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    return texID;
}

std::unique_ptr<Texture>
OpenGLDevice::createCubemapTexture(const std::string& crossPath) {
    const GLuint handle = loadCubemapCross(crossPath);
    if (!handle)
        return nullptr;
    GLint width = 0, height = 0;
    glBindTexture(GL_TEXTURE_CUBE_MAP, handle);
    glGetTexLevelParameteriv(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0,
                             GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0,
                             GL_TEXTURE_HEIGHT, &height);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    return std::make_unique<OpenGLTexture>(handle, width, height);
}

std::unique_ptr<Texture> OpenGLDevice::createCubemapTexture(
    const std::vector<std::string>& facePaths) {
    const GLuint handle = loadCubemap(facePaths);
    if (!handle)
        return nullptr;
    GLint width = 0, height = 0;
    glBindTexture(GL_TEXTURE_CUBE_MAP, handle);
    glGetTexLevelParameteriv(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0,
                             GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0,
                             GL_TEXTURE_HEIGHT, &height);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    return std::make_unique<OpenGLTexture>(handle, width, height);
}

// Unit-cube VAO for skybox rendering
GLuint OpenGLDevice::makeSkyboxVAO() {
    static const float verts[] = {
        -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, // Back face vertices
        -1, -1, 1,  1, -1, 1,  1, 1, 1,  -1, 1, 1   // Front face vertices
    };
    static const unsigned int indices[] = {
        0, 1, 2, 2, 3, 0, // Front (looking from inside)
        1, 5, 6, 6, 2, 1, // Right
        5, 4, 7, 7, 6, 5, // Back
        4, 0, 3, 3, 7, 4, // Left
        3, 2, 6, 6, 7, 3, // Top
        4, 5, 1, 1, 0, 4  // Bottom
    };
    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glBindVertexArray(0);
    return vao;
}

void OpenGLDevice::applySkyboxTex(GLuint tex, UpAxis upAxis) {
    _skyboxUpAxis = upAxis;
    if (_skyboxTex)
        glDeleteTextures(1, &_skyboxTex);
    if (!_skyboxVAO)
        _skyboxVAO = makeSkyboxVAO();
    if (!_skyboxShader)
        _skyboxShader = createShader(skyboxVs, skyboxFs);
    _skyboxTex = tex;
    _skyboxShader->use();
    _skyboxShader->setInt("skybox", 0);
}

void OpenGLDevice::setSkybox(const std::string& path, UpAxis upAxis) {
    applySkyboxTex(loadCubemapCross(path), upAxis);
}

void OpenGLDevice::setSkybox(const std::vector<std::string>& paths,
                             UpAxis upAxis) {
    applySkyboxTex(loadCubemap(paths), upAxis);
}

void OpenGLDevice::drawSkybox(const glm::mat4& view, const glm::mat4& proj) {
    if (!_skyboxTex || !_skyboxShader)
        return;
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);
    _skyboxShader->use();
    _skyboxShader->setMat4("view", view);
    _skyboxShader->setMat4("projection", proj);
    _skyboxShader->setBool("zUp", _skyboxUpAxis == UpAxis::Z);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, _skyboxTex);
    glBindVertexArray(_skyboxVAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
}

// End Skybox
// ---------------------------------------------------------------------------

} // namespace Backend
} // namespace KE
