///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include <glm/glm.hpp>
#include <vector>

#include "utils/types.hpp"

namespace KE {

struct WorldFrustumPos {
    glm::vec3 nearLB = {-1., -1., -1.};
    glm::vec3 nearLT = {-1., +1., -1.};
    glm::vec3 nearRB = {+1., -1., -1.};
    glm::vec3 nearRT = {+1., +1., -1.};
    glm::vec3 farLB = {-1., -1., +1.};
    glm::vec3 farLT = {-1., +1., +1.};
    glm::vec3 farRB = {+1., -1., +1.};
    glm::vec3 farRT = {+1., +1., +1.};
};

class Camera {

  private:
    glm::vec3 _cameraPos, _targetPos, _cameraLookDir, _cameraUpDir,
        _cameraRightDir;
    glm::mat4 _viewMatrix, _projMatrix;
    glm::mat4 _invViewMatrix, _invProjMatrix, _invViewProjMatrix;
    glm::vec3 _upVector;
    UpAxis _upAxis;
    float _FoV, _camToLookDistance, _nearPlane, _farPlane;
    int _screenWidth, _screenHeight;

  public:
    float pole, azimuth;
    Camera();
    Camera(glm::vec3 cameraPos, glm::vec3 targetPos, UpAxis upAxis);
    ~Camera();
    void init(glm::vec3 cameraPos, glm::vec3 targetPos, UpAxis upAxis);
    glm::vec3 getCameraPos() { return _cameraPos; };
    glm::vec3 getTargetPos() { return _targetPos; };
    glm::mat4 getViewMatrix() { return _viewMatrix; }
    glm::mat4 getProjMatrix() { return _projMatrix; }
    glm::mat4 getInvViewMatrix() { return _invViewMatrix; }
    glm::mat4 getInvProjMatrix() { return _invProjMatrix; }
    glm::mat4 getInvViewProjMatrix() { return _invViewProjMatrix; }
    void updateViewMatrix();
    void updateProjMatrix(const int scrWidth, const int scrHeight);

    void setCameraPos(glm::vec3 cameraPos);
    void setTargetPos(glm::vec3 targetPos);
    void setFoV(float FoV);
    void setNearPlane(const float dis);
    void setFarPlane(const float dis);

    float getFoV() { return _FoV; }
    float getNearPlane() { return _nearPlane; }
    float getFarPlane() { return _farPlane; }
    float getCamToLookDistance();

    UpAxis getUpAxis() const { return _upAxis; }
    glm::vec3 getUpVector() const { return _upVector; }

    glm::vec3 getCameraLookDir() { return _cameraLookDir; };
    glm::vec3 getCameraUpDir() { return _cameraUpDir; };
    glm::vec3 getCameraRightDir() { return _cameraRightDir; }
    glm::vec3 ndcToWorld(glm::vec3 ndc);
    WorldFrustumPos getFrustumPos();
    WorldFrustumPos getFrustumPos(float nearPlane, float farPlane);
};

} // namespace KE

#endif
