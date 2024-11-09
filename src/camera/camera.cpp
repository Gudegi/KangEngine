#include "camera.hpp"
#include <glm/gtc/matrix_transform.hpp>

void Camera::updateViewMatrix()
{
    _viewMatrix = glm::translate(glm::mat4(1.0f), _cameraPos);
    glm::vec3 cameraLookAtDir = glm::normalize(_targetPos - _cameraPos); // z
    glm::vec3 cameraRightDir = glm::normalize(glm::cross(cameraLookAtDir, _upAxis)); // x
    glm::vec3 cameraUpDir = glm::cross(cameraRightDir, cameraLookAtDir); // y
    _viewMatrix[0] = glm::vec4(cameraRightDir, 0.0f);
    _viewMatrix[1] = glm::vec4(cameraUpDir, 0.0f);
    _viewMatrix[2] = glm::vec4(cameraLookAtDir, 0.0f);
}

Camera::Camera(glm::vec3 cameraPos, glm::vec3 targetPos, char upAxis): _cameraPos(cameraPos), _targetPos(targetPos)
{
    if (upAxis == 'y')
    {
        _upAxis = glm::vec3(0.0, 1.0, 0.0f);
    }
    else if (upAxis == 'z')
    {
        _upAxis = glm::vec3(0.0, 0.0, 1.0f);
    }
    this->updateViewMatrix();
}

Camera::~Camera()
{

}

glm::vec3 Camera::getCameraPos()
{
    return _cameraPos;
}

glm::vec3 Camera::getTargetPos()
{
    return _targetPos;
}

glm::mat4 Camera::getViewMatrix()
{
    return _viewMatrix;
}

void Camera::setCameraPos(glm::vec3 cameraPos)
{
    _cameraPos = cameraPos;
    updateViewMatrix();
}

void Camera::setTargetPos(glm::vec3 targetPos)
{
    _targetPos = targetPos;
    updateViewMatrix();
}


