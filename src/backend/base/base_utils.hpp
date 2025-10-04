#ifndef _BASE_UTILS_HPP_
#define _BASE_UTILS_HPP_

#include "graphics_device.hpp"

namespace KE {
namespace Backend {

TextureDesc loadImage(const std::string path, bool flip=false);

} // namespace Backend
} // namespace KE

#endif
