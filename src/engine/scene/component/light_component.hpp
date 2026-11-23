#ifndef _SCENE_LIGHT_COMPONENT_HPP_
#define _SCENE_LIGHT_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"
#include "engine/scene/types/light.hpp"

namespace KE {
namespace Scene {

// Owns light settings and local emission direction.
// Point/spot positions use Prim translation, relative to the parent.
// Directions point along emitted rays and rotate with the Prim.
class LightComponent : public ComponentBase {
  public:
    LightType type() const;

    void setDirectionalLight(const DirectionalLight& light);
    DirectionalLight directionalLight() const;

    void setPointLight(const PointLight& light);
    PointLight pointLight() const;

    void setSpotLight(const SpotLight& light);
    SpotLight spotLight() const;

    void setRectLight(const RectLight& light);
    RectLight rectLight() const;
    void setDomeLight(const DomeLight& light);
    DomeLight domeLight() const;

  private:
    friend class Prim;

    explicit LightComponent(Prim* owner);
    void detach();

    LightType _type = LightType::Point;
    RectLight _rect;
    DomeLight _dome;
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
