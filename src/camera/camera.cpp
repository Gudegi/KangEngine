#include "camera.hpp"
#include <glm/gtc/matrix_transform.hpp>

void Camera::updateViewMatrix()
{   
    _cameraLookDir = glm::normalize(_targetPos - _cameraPos); // z
    _cameraRightDir = glm::normalize(glm::cross(_cameraLookDir, _upAxis)); // x
    _cameraUpDir = glm::cross(_cameraRightDir, _cameraLookDir); // y
    
    /* // same but slow.
    _viewMatrix = glm::translate(glm::mat4(1.0f), _cameraPos);
    _viewMatrix[0] = glm::vec4(_cameraRightDir, 0.0f);
    _viewMatrix[1] = glm::vec4(_cameraUpDir, 0.0f);
    _viewMatrix[2] = glm::vec4(-_cameraLookDir, 0.0f);
    _viewMatrix = glm::inverse(_viewMatrix);
    */

    _viewMatrix = glm::mat4(1.0f);
    _viewMatrix[0][0] = _cameraRightDir.x;
    _viewMatrix[1][0] = _cameraRightDir.y;
    _viewMatrix[2][0] = _cameraRightDir.z;
    _viewMatrix[0][1] = _cameraUpDir.x;
    _viewMatrix[1][1] = _cameraUpDir.y;
    _viewMatrix[2][1] = _cameraUpDir.z;
    _viewMatrix[0][2] = -_cameraLookDir.x;
    _viewMatrix[1][2] = -_cameraLookDir.y;
    _viewMatrix[2][2] = -_cameraLookDir.z;
    _viewMatrix[3][0] = -glm::dot(_cameraRightDir, _cameraPos);
    _viewMatrix[3][1] = -glm::dot(_cameraUpDir, _cameraPos);
    _viewMatrix[3][2] = glm::dot(_cameraLookDir, _cameraPos);
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

glm::vec3 Camera::getCameraLookDir()
{
    return _cameraLookDir;
}
glm::vec3 Camera::getCameraUpDir()
{
    return _cameraUpDir;
}
glm::vec3 Camera::getCameraRightDir()
{
    return _cameraRightDir;
}

