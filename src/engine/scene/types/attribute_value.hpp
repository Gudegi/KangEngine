#pragma once

// Values used by Prim metadata and TransformComponent's authored operations.
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <string>
#include <variant>
#include <vector>

namespace KE {
namespace Scene {

using AttributeValue =
    std::variant<bool, int, float, std::string, glm::vec3, glm::vec4, glm::mat4,
                 glm::quat, std::vector<std::string>>;

} // namespace Scene
} // namespace KE
