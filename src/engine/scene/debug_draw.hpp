#ifndef _DEBUG_DRAW_HPP_
#define _DEBUG_DRAW_HPP_

#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/scene/scene_backend.hpp"
#include <cstddef>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>

namespace KE {
class App;
namespace Backend {
class Shader;
}
namespace Scene {

class DebugDraw {
  public:
    static bool makeArrowTransform(const glm::vec3& start,
                                   const glm::vec3& end,
                                   glm::mat4& transform);
    static bool makeLineTransform(const glm::vec3& start,
                                  const glm::vec3& end,
                                  glm::mat4& transform);

    // Scene-backed debug lines. Useful when the caller wants explicit prims.
    static std::vector<Prim*>
    logLines(SceneBackend* scene, const std::string& basePath,
             const std::vector<glm::vec3>& starts,
             const std::vector<glm::vec3>& ends,
             const std::vector<glm::vec4>& colors = {},
             float radius = 0.005f, int segments = 8);

    static std::vector<Prim*>
    logArrows(SceneBackend* scene, const std::string& basePath,
              const std::vector<glm::vec3>& starts,
              const std::vector<glm::vec3>& ends,
              const std::vector<glm::vec4>& colors = {},
              float radius = 0.02f, int segments = 12);

    static std::vector<Prim*>
    logCoordinateAxes(SceneBackend* scene, const std::string& basePath,
                      glm::vec3 origin = glm::vec3(0.0f),
                      glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f,
                                                        0.0f),
                      float length = 1.0f, float radius = 0.005f,
                      int segments = 8);

    // Instanced debug lines, closer to Newton viewer.log_lines(). Debug
    // geometry is excluded from the shadow pass.
    static MeshHandle
    logLines(App* app, Backend::Shader* shader, const std::string& path,
             const std::vector<glm::vec3>& starts,
             const std::vector<glm::vec3>& ends,
             const std::vector<glm::vec4>& colors = {},
             float radius = 0.005f, int segments = 8);

    static MeshHandle logLines(App* app, Backend::Shader* shader,
                               const std::string& path, const float* starts,
                               const float* ends, const float* colors,
                               size_t count, size_t colorCount,
                               float radius = 0.005f, int segments = 8);

    static void updateLines(App* app, MeshHandle handle,
                            const std::vector<glm::vec3>& starts,
                            const std::vector<glm::vec3>& ends,
                            const std::vector<glm::vec4>& colors = {});

    static void updateLines(App* app, MeshHandle handle, const float* starts,
                            const float* ends, const float* colors,
                            size_t count, size_t colorCount);

    static MeshHandle
    logArrows(App* app, Backend::Shader* shader, const std::string& path,
              const std::vector<glm::vec3>& starts,
              const std::vector<glm::vec3>& ends,
              const std::vector<glm::vec4>& colors = {},
              float radius = 0.02f, int segments = 12);

    static MeshHandle logArrows(App* app, Backend::Shader* shader,
                                const std::string& path, const float* starts,
                                const float* ends, const float* colors,
                                size_t count, size_t colorCount,
                                float radius = 0.02f, int segments = 12);

    static void updateArrows(App* app, MeshHandle handle,
                             const std::vector<glm::vec3>& starts,
                             const std::vector<glm::vec3>& ends,
                             const std::vector<glm::vec4>& colors = {});

    static void updateArrows(App* app, MeshHandle handle, const float* starts,
                             const float* ends, const float* colors,
                             size_t count, size_t colorCount);

    static MeshHandle
    logCoordinateAxes(App* app, Backend::Shader* shader,
                      const std::string& path,
                      glm::vec3 origin = glm::vec3(0.0f),
                      glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f,
                                                        0.0f),
                      float length = 1.0f, float radius = 0.005f,
                      int segments = 8);
};

} // namespace Scene
} // namespace KE

#endif
