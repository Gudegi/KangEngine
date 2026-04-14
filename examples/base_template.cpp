
#include "kangEngine.hpp"
#include <iostream>
#include <memory>

using namespace KE;

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

class MyApp : public App {
  public:
    void initialize(int width, int height, Backend::BackendType backendType) {
        App::initialize(width, height, false, UpAxis::Y, backendType);
    }

    void setup() override {}

    void preRender() override {}

    void render() override {}

    void postRender() override {}
};

int main() {
    MyApp app;
    Backend::BackendType backend = Backend::BackendType::OpenGL;
    app.initialize(SCR_WIDTH, SCR_HEIGHT, backend);
    app.start();
    return 0;
}
