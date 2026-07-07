#ifndef _SCENE_LIGHT_COMPONENT_HPP_
#define _SCENE_LIGHT_COMPONENT_HPP_

#include "engine/scene/light.hpp"

#include <cstdint>

namespace KE {
namespace Scene {

class Prim;

// Runtime light state attached to a Light prim. Prim owns identity/path/transform;
// LightComponent owns renderer-independent light data.
class LightComponent {
  public:
    LightComponent(const LightComponent&) = delete;
    LightComponent& operator=(const LightComponent&) = delete;

    bool isAttached() const { return _owner != nullptr; }
    Prim* owner() const { return _owner; }

    LightType type() const;

    uint64_t version() const { return _version; }

    void setDirectionalLight(const DirectionalLight& light);
    DirectionalLight directionalLight() const;

    void setPointLight(const PointLight& light);
    PointLight pointLight() const;

    void setSpotLight(const SpotLight& light);
    SpotLight spotLight() const;

  private:
    friend class Prim;

    explicit LightComponent(Prim* owner);
    void detach();
    void requireAttached() const;
    void markChanged();

    Prim* _owner = nullptr;
    LightType _type = LightType::Point;
    glm::vec3 _direction = {0.0f, 0.0f, -1.0f};
    glm::vec3 _color = {1.0f, 1.0f, 1.0f};
    float _intensity = 1.0f;
    glm::vec3 _ambient = {0.15f, 0.15f, 0.15f};
    float _range = 10.0f;
    float _innerConeAngle = glm::radians(20.0f);
    float _outerConeAngle = glm::radians(30.0f);
    uint64_t _version = 1;
};

} // namespace Scene
} // namespace KE

#endif
