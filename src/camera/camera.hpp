///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include <glm/glm.hpp>
#include <vector>

namespace KE {

class Camera
{

private:
    glm::vec3 _cameraPos, _targetPos, _cameraLookDir, _cameraUpDir, _cameraRightDir;
    glm::mat4 _viewMatrix, _projMatrix;
    glm::vec3 _upAxis;
    float _FoV, _camToLookDistance, _nearPlane, _farPlane;

public:
    Camera();
    Camera(glm::vec3 cameraPos, glm::vec3 targetPos, char upAxis);
    ~Camera();
    void init(glm::vec3 cameraPos, glm::vec3 targetPos, char upAxis);
    glm::vec3 getCameraPos();
    glm::vec3 getTargetPos();
    glm::mat4 getViewMatrix();
    glm::mat4 getProjMatrix();
    void updateViewMatrix();
    void updateProjMatrix(const unsigned int scrWidth, const unsigned int scrHeight);

    void setCameraPos(glm::vec3 cameraPos);
    void setTargetPos(glm::vec3 targetPos);
    void setFoV(const float FoV);
    void setNearPlane(const float dis);
    void setFarPlane(const float dis);

    float getFoV() {return _FoV;}
    float getNearPlane() {return _nearPlane;}
    float getFarPlane() {return _farPlane;}
    float getCamToLookDistance();
    
    glm::vec3 calcSpherePos();
    glm::vec3 getCameraLookDir();
    glm::vec3 getCameraUpDir();
    glm::vec3 getCameraRightDir();

};

} // namespace KE

#endif