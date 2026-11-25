// Checks framebuffer size before any event polling or rendering.
#include "engine/core/window/window.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

int main() {
    const std::array<std::array<int, 2>, 3> sizes = {
        {{128, 96}, {256, 256}, {640, 480}}};
    for (int run = 0; run < 20; ++run) {
        const auto& size = sizes[run % sizes.size()];
        KE::Window window;
        window.init(size[0], size[1], true);
        int width = 0, height = 0;
        glfwGetFramebufferSize(window.getGlfwWindow(), &width, &height);
        if (width != size[0] || height != size[1]) {
            std::cerr << "Requested " << size[0] << 'x' << size[1] << ", got "
                      << width << 'x' << height << '\n';
            throw std::runtime_error("initial framebuffer size is stale");
        }
    }
    std::cout << "PASS: initial framebuffer size without event polling\n";
}
