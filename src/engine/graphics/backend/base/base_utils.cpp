#include "base_utils.hpp"
#include "graphics_device.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <iostream>
#include <memory>

namespace KE {
namespace Backend {

TextureDesc loadImage(const std::string path, bool flip)
{
    TextureDesc desc;
    desc.name = path;

    stbi_set_flip_vertically_on_load(flip);

    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load texture '" << path << "': " 
                  << stbi_failure_reason() << std::endl;
        stbi_image_free((void*)data);    
        return desc;  // return empty descriptor
    }
    else {
        desc.width = width;
        desc.height = height;
        desc.channels = channels;
        desc.data = data;
        //stbi_image_free((void*)desc.data) // NOTE: You need release this data memory in device specific texture constructor.
        return desc;
    }
}

} // namespace Backend
} // namespace KE

