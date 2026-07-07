#ifndef _SCENE_CAMERA_COMPONENT_HPP_
#define _SCENE_CAMERA_COMPONENT_HPP_

#include <cstdint>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace KE {
namespace Scene {

class Prim;

enum class CameraProjectionType {
    Perspective,
    Orthographic,
};

// Runtime authored camera state attached to a Camera prim. Prim owns
// identity/path/transform; CameraComponent owns projection settings.
class CameraComponent {
  public:
    CameraComponent(const CameraComponent&) = delete;
    CameraComponent& operator=(const CameraComponent&) = delete;

    bool isAttached() const { return _owner != nullptr; }
    Prim* owner() const { return _owner; }

    CameraProjectionType projectionType() const;
    uint64_t version() const { return _version; }

    void setPerspective(float verticalFovDegrees, float nearPlane,
                        float farPlane);
    void setOrthographic(float verticalSize, float nearPlane, float farPlane);

    float verticalFovDegrees() const;
    float orthographicSize() const;
    float nearPlane() const;
    float farPlane() const;

    glm::vec3 position() const;
    glm::vec3 forward() const;
    glm::vec3 up() const;
    glm::mat4 viewMatrix() const;
    glm::mat4 projectionMatrix(float aspect) const;
    glm::mat4 viewProjectionMatrix(float aspect) const;

  private:
    friend class Prim;

    explicit CameraComponent(Prim* owner);
    void detach();
    void requireAttached() const;
    void markChanged();

    Prim* _owner = nullptr;
    CameraProjectionType _projectionType = CameraProjectionType::Perspective;
    float _verticalFovDegrees = 45.0f;
    float _orthographicSize = 5.0f;
    float _nearPlane = 0.1f;
    float _farPlane = 100.0f;
    uint64_t _version = 1;
};

} // namespace Scene
} // namespace KE

#endif
