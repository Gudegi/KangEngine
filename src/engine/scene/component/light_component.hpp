#ifndef _SCENE_LIGHT_COMPONENT_HPP_
#define _SCENE_LIGHT_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"
#include "engine/scene/light.hpp"

namespace KE {
namespace Scene {

// Runtime light state attached to a Light prim. Prim owns identity/path/transform;
// LightComponent owns renderer-independent light data.
class LightComponent : public ComponentBase {
  public:
    LightType type() const;

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

    LightType _type = LightType::Point;
    glm::vec3 _direction = {0.0f, 0.0f, -1.0f};
    glm::vec3 _color = {1.0f, 1.0f, 1.0f};
    float _intensity = 1.0f;
    glm::vec3 _ambient = {0.15f, 0.15f, 0.15f};
    float _range = 10.0f;
    float _innerConeAngle = glm::radians(20.0f);
    float _outerConeAngle = glm::radians(30.0f);
};

} // namespace Scene
} // namespace KE

#endif
