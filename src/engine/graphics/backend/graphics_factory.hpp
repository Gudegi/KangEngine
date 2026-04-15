///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _GRAPHICS_FACTORY_HPP_
#define _GRAPHICS_FACTORY_HPP_

#include "base/graphics_device.hpp"
#include "opengl/opengl_device.hpp"
#include <memory>
#include <iostream>

namespace KE {
namespace Backend {

class GraphicsFactory {
public:
    static std::unique_ptr<GraphicsDevice> createDevice(BackendType type) {
        switch(type) {
            case BackendType::OpenGL:
                return std::make_unique<OpenGLDevice>();
            case BackendType::Vulkan:
                // TODO: Implement VulkanDevice
                std::cerr << "Vulkan backend not implemented yet" << std::endl;
                return nullptr;
            case BackendType::WebGPU:
                // TODO: Implement WebGPUDevice
                std::cerr << "WebGPU backend not implemented yet" << std::endl;
                return nullptr;
            default:
                std::cerr << "Unknown backend type" << std::endl;
                return nullptr;
        }
    }

    static BackendType getDefaultBackend() {
        return BackendType::OpenGL;
    }

    static const char* getBackendName(BackendType type) {
        switch(type) {
            case BackendType::OpenGL: return "OpenGL";
            case BackendType::Vulkan: return "Vulkan";
            case BackendType::WebGPU: return "WebGPU";
            default: return "Unknown";
        }
    }
};

} // namespace Backend
} // namespace KE

#endif