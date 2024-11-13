#include "camera.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

Camera::Camera(glm::vec3 cameraPos, glm::vec3 targetPos, char upAxis): _cameraPos(cameraPos), _targetPos(targetPos)
{
     glm::vec3 tmpVec3 = (_targetPos - _cameraPos);
    _camToLookDistance = sqrt(pow(tmpVec3.x, 2) + pow(tmpVec3.y, 2) + pow(tmpVec3.z, 2));
    _FoV = 45.0f;
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

Camera::Camera()
{
}


Camera::~Camera()
{

}

void Camera::init(glm::vec3 cameraPos, glm::vec3 targetPos, char upAxis)
{   
    _cameraPos = cameraPos;
    _targetPos = targetPos;
    glm::vec3 tmpVec3 = (targetPos - cameraPos);
    _camToLookDistance = sqrt(pow(tmpVec3.x, 2) + pow(tmpVec3.y, 2) + pow(tmpVec3.z, 2));
    _FoV = 45.0f;
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

glm::vec3 Camera::getCameraPos()
{
    return _cameraPos;
}

glm::vec3 Camera::getTargetPos()
{
    return _targetPos;
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

glm::mat4 Camera::getViewMatrix()
{
    return _viewMatrix;
}

glm::mat4 Camera::getProjMatrix(const unsigned int scrWidth, const unsigned int scrHeight, const float zNear, const float zFar)
{   
    return glm::perspective(glm::radians(_FoV), (float)scrWidth / (float)scrHeight, zNear, zFar);
}

void Camera::setFoV(const float FoV)
{
    _FoV = FoV;
}

float Camera::getFoV()
{
    return _FoV;    
}

float Camera::getCamToLookDistance()
{
    return _camToLookDistance;
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

