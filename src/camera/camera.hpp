///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include <glm/glm.hpp>

class Camera
{

private:
    glm::vec3 _cameraPos, _targetPos;
    glm::mat4 _viewMatrix;
    glm::vec3 _upAxis;
    int _FOV;
    void updateViewMatrix();

public:
    Camera(glm::vec3 cameraPos, glm::vec3 targetPos, char upAxis='y');
    ~Camera();
    glm::vec3 getCameraPos();
    glm::vec3 getTargetPos();
    glm::mat4 getViewMatrix();
    void setCameraPos(glm::vec3 cameraPos);
    void setTargetPos(glm::vec3 targetPos);


};

#endif