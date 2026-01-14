#include "camera.hpp"
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

namespace KE {

Camera::Camera(glm::vec3 cameraPos, glm::vec3 targetPos, char upAxis)
    : _cameraPos(cameraPos), _targetPos(targetPos) {
    _camToLookDistance = glm::length(_targetPos - _cameraPos);
    _FoV = 45.0f;
    _nearPlane = 0.1f;
    _farPlane = 100.0f;
    _screenWidth = 1920;
    _screenHeight = 1080;

    if (upAxis == 'y') {
        _upAxis = glm::vec3(0.0f, 1.0f, 0.0f);
    } else if (upAxis == 'z') {
        _upAxis = glm::vec3(0.0f, 0.0f, 1.0f);
    } else {
        // Default to Y-up if invalid axis provided
        _upAxis = glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // calc pole, azimuth from curr pos
    glm::vec3 offset = cameraPos - targetPos;
    float r = glm::length(offset);

    if (r > 0.0001f) {
        // pole: Y (0deg up, 180deg down)
        pole = glm::degrees(glm::acos(offset.y / r));
        // azimuth: XZ
        azimuth = glm::degrees(glm::atan(offset.z, offset.x));
    } else {
        pole = 90.0f;
        azimuth = 0.0f;
    }

    this->updateViewMatrix();
}

Camera::Camera() {}

Camera::~Camera() {}

void Camera::init(glm::vec3 cameraPos, glm::vec3 targetPos, char upAxis) {
    _cameraPos = cameraPos;
    _targetPos = targetPos;
    _camToLookDistance = glm::length(targetPos - cameraPos);
    _FoV = 45.0f;
    _nearPlane = 0.1f;
    _farPlane = 100.0f;
    _screenWidth = 1920;
    _screenHeight = 1080;

    if (upAxis == 'y') {
        _upAxis = glm::vec3(0.0f, 1.0f, 0.0f);
    } else if (upAxis == 'z') {
        _upAxis = glm::vec3(0.0f, 0.0f, 1.0f);
    } else {
        // Default to Y-up if invalid axis provided
        _upAxis = glm::vec3(0.0f, 1.0f, 0.0f);
    }

    glm::vec3 offset = cameraPos - targetPos;
    float r = glm::length(offset);

    if (r > 0.0001f) {
        pole = glm::degrees(glm::acos(offset.y / r));
        azimuth = glm::degrees(glm::atan(offset.z, offset.x));
    } else {
        pole = 90.0f;
        azimuth = 0.0f;
    }

    this->updateViewMatrix();
}

void Camera::updateViewMatrix() {
    /*
    _cameraLookDir = glm::normalize(_targetPos - _cameraPos); // z
    _cameraRightDir = glm::normalize(glm::cross(_cameraLookDir, _upAxis)); // x
    _cameraUpDir = glm::cross(_cameraRightDir, _cameraLookDir); // y
    */

    /* // same but slow.
    _viewMatrix = glm::translate(glm::mat4(1.0f), _cameraPos);
    _viewMatrix[0] = glm::vec4(_cameraRightDir, 0.0f);
    _viewMatrix[1] = glm::vec4(_cameraUpDir, 0.0f);
    _viewMatrix[2] = glm::vec4(-_cameraLookDir, 0.0f);
    _viewMatrix = glm::inverse(_viewMatrix);
    */

    glm::vec3 lookVec = _targetPos - _cameraPos;
    // Check for zero vector (camera and target at same position)
    if (glm::length(lookVec) < 0.0001f) {
        // Use a default forward direction if positions are the same
        lookVec = glm::vec3(0.0f, 0.0f, -1.0f);
    }
    _cameraLookDir = glm::normalize(lookVec);

    // Check for gimbal lock (camera looking parallel to up axis)
    glm::vec3 crossResult = glm::cross(_cameraLookDir, _upAxis);
    if (glm::length(crossResult) < 0.0001f) {
        // Camera is looking straight up or down, use alternative right vector
        glm::vec3 altRight = glm::vec3(1.0f, 0.0f, 0.0f);
        if (glm::abs(glm::dot(_cameraLookDir, altRight)) > 0.99f) {
            altRight = glm::vec3(0.0f, 0.0f, 1.0f);
        }
        _cameraRightDir = glm::normalize(glm::cross(_cameraLookDir, altRight));
    } else {
        _cameraRightDir = glm::normalize(crossResult);
    }

    _cameraUpDir = glm::cross(_cameraRightDir, _cameraLookDir);

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

glm::vec3 Camera::getCameraPos() { return _cameraPos; }

glm::vec3 Camera::getTargetPos() { return _targetPos; }

glm::vec3 Camera::getCameraLookDir() { return _cameraLookDir; }
glm::vec3 Camera::getCameraUpDir() { return _cameraUpDir; }
glm::vec3 Camera::getCameraRightDir() { return _cameraRightDir; }

glm::mat4 Camera::getViewMatrix() { return _viewMatrix; }

void Camera::updateProjMatrix(const unsigned int scrWidth,
                              const unsigned int scrHeight) {
    _screenWidth = scrWidth;
    _screenHeight = scrHeight;

    // Prevent division by zero
    if (_screenHeight == 0) {
        _projMatrix =
            glm::perspective(glm::radians(_FoV), 1.0f, _nearPlane, _farPlane);
        return;
    }

    _projMatrix = glm::perspective(glm::radians(_FoV),
                                   (float)_screenWidth / (float)_screenHeight,
                                   _nearPlane, _farPlane);
}

glm::mat4 Camera::getProjMatrix() { return _projMatrix; }

void Camera::setFoV(float FoV) {
    FoV = std::clamp(FoV, 10.f, 120.0f);
    _FoV = FoV;
    updateProjMatrix(_screenWidth, _screenHeight);
}

void Camera::setNearPlane(const float dis) {
    _nearPlane = dis;
    updateProjMatrix(_screenWidth, _screenHeight);
}

void Camera::setFarPlane(const float dis) {
    _farPlane = dis;
    updateProjMatrix(_screenWidth, _screenHeight);
}

float Camera::getCamToLookDistance() { return _camToLookDistance; }

void Camera::setCameraPos(glm::vec3 cameraPos) {
    _cameraPos = cameraPos;
    _camToLookDistance = glm::length(_targetPos - _cameraPos);
    updateViewMatrix();
}

void Camera::setTargetPos(glm::vec3 targetPos) {
    _targetPos = targetPos;
    _camToLookDistance = glm::length(_targetPos - _cameraPos);
    updateViewMatrix();
}

} // namespace KE
