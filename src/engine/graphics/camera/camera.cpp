#include "camera.hpp"
#include <algorithm>
#include <cmath>
#include <glm/fwd.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace KE {

Camera::Camera(glm::vec3 cameraPos, glm::vec3 targetPos, UpAxis upAxis)
    : _cameraPos(cameraPos), _targetPos(targetPos),
      _cameraLookDir(0.0f, 0.0f, -1.0f), _cameraUpDir(0.0f, 1.0f, 0.0f),
      _cameraRightDir(1.0f, 0.0f, 0.0f), _viewMatrix(1.0f), _projMatrix(1.0f),
      _invViewMatrix(1.0f), _invProjMatrix(1.0f), _invViewProjMatrix(1.0f),
      _upAxis(upAxis) {
    _camToLookDistance = glm::length(_targetPos - _cameraPos);
    _FoV = 45.0f;
    _nearPlane = 0.1f;
    _farPlane = 100.0f;
    _screenWidth = 1920;
    _screenHeight = 1080;

    _upVector = (_upAxis == UpAxis::Z) ? glm::vec3(0.0f, 0.0f, 1.0f)
                                       : glm::vec3(0.0f, 1.0f, 0.0f);

    // calc pole, azimuth from curr pos
    glm::vec3 offset = cameraPos - targetPos;
    float r = glm::length(offset);

    if (r > 0.0001f) {
        if (_upAxis == UpAxis::Z) {
            pole = glm::degrees(glm::acos(offset.z / r));
            azimuth = glm::degrees(glm::atan(offset.y, offset.x));
        } else {
            // pole: Y (0deg up, 180deg down)
            pole = glm::degrees(glm::acos(offset.y / r));
            // azimuth: XZ
            azimuth = glm::degrees(glm::atan(offset.z, offset.x));
        }
    } else {
        pole = 90.0f;
        azimuth = 0.0f;
    }

    this->updateProjMatrix(_screenWidth, _screenHeight);
    this->updateViewMatrix();
}

Camera::Camera()
    : _cameraPos(0.0f, 0.0f, 1.0f), _targetPos(0.0f, 0.0f, 0.0f),
      _cameraLookDir(0.0f, 0.0f, -1.0f), _cameraUpDir(0.0f, 1.0f, 0.0f),
      _cameraRightDir(1.0f, 0.0f, 0.0f), _viewMatrix(1.0f), _projMatrix(1.0f),
      _invViewMatrix(1.0f), _invProjMatrix(1.0f), _invViewProjMatrix(1.0f),
      _upVector(0.0f, 1.0f, 0.0f), _upAxis(UpAxis::Y), _FoV(45.0f),
      _camToLookDistance(1.0f), _nearPlane(0.1f), _farPlane(100.0f),
      _screenWidth(1920), _screenHeight(1080), pole(90.0f), azimuth(0.0f) {
    updateProjMatrix(_screenWidth, _screenHeight);
    updateViewMatrix();
}

Camera::~Camera() {}

void Camera::init(glm::vec3 cameraPos, glm::vec3 targetPos, UpAxis upAxis) {
    _cameraPos = cameraPos;
    _targetPos = targetPos;
    _upAxis = upAxis;
    _camToLookDistance = glm::length(targetPos - cameraPos);
    _FoV = 45.0f;
    _nearPlane = 0.1f;
    _farPlane = 100.0f;
    _screenWidth = 1920;
    _screenHeight = 1080;

    _upVector = (_upAxis == UpAxis::Z) ? glm::vec3(0.0f, 0.0f, 1.0f)
                                       : glm::vec3(0.0f, 1.0f, 0.0f);

    glm::vec3 offset = cameraPos - targetPos;
    float r = glm::length(offset);

    if (r > 0.0001f) {
        if (_upAxis == UpAxis::Z) {
            pole = glm::degrees(glm::acos(offset.z / r));
            azimuth = glm::degrees(glm::atan(offset.y, offset.x));
        } else {
            pole = glm::degrees(glm::acos(offset.y / r));
            azimuth = glm::degrees(glm::atan(offset.z, offset.x));
        }
    } else {
        pole = 90.0f;
        azimuth = 0.0f;
    }

    this->updateProjMatrix(_screenWidth, _screenHeight);
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
    glm::vec3 crossResult = glm::cross(_cameraLookDir, _upVector);
    if (glm::length(crossResult) < 0.0001f) {
        // Camera is looking straight up or down, use alternative right vector
        glm::vec3 altRight = glm::vec3(1.0f, 0.0f, 0.0f);
        if (glm::abs(glm::dot(_cameraLookDir, altRight)) > 0.99f) {
            altRight = (_upAxis == UpAxis::Z) ? glm::vec3(0.0f, 1.0f, 0.0f)
                                              : glm::vec3(0.0f, 0.0f, 1.0f);
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

    _invViewMatrix = glm::inverse(_viewMatrix);
    _invViewProjMatrix = glm::inverse(_projMatrix * _viewMatrix);
}

void Camera::updateProjMatrix(const int scrWidth, const int scrHeight) {
    _screenWidth = scrWidth;
    _screenHeight = scrHeight;

    if (_screenHeight == 0) {
        _projMatrix =
            glm::perspective(glm::radians(_FoV), 1.0f, _nearPlane, _farPlane);
        _invProjMatrix = glm::inverse(_projMatrix);
        _invViewProjMatrix = glm::inverse(_projMatrix * _viewMatrix);
        return;
    }

    _projMatrix = glm::perspective(glm::radians(_FoV),
                                   (float)_screenWidth / (float)_screenHeight,
                                   _nearPlane, _farPlane);
    _invProjMatrix = glm::inverse(_projMatrix);
    _invViewProjMatrix = glm::inverse(_projMatrix * _viewMatrix);
}

void Camera::setFoV(float FoV) {
    FoV = std::clamp(FoV, 10.f, 90.0f);
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
    glm::vec3 offset = _cameraPos - _targetPos;
    float r = glm::length(offset);
    if (r > 0.0001f) {
        if (_upAxis == UpAxis::Z) {
            pole = glm::degrees(glm::acos(glm::clamp(offset.z / r, -1.f, 1.f)));
            azimuth = glm::degrees(glm::atan(offset.y, offset.x));
        } else {
            pole = glm::degrees(glm::acos(glm::clamp(offset.y / r, -1.f, 1.f)));
            azimuth = glm::degrees(glm::atan(offset.z, offset.x));
        }
    }
    updateViewMatrix();
}

void Camera::setTargetPos(glm::vec3 targetPos) {
    _targetPos = targetPos;
    _camToLookDistance = glm::length(_targetPos - _cameraPos);
    updateViewMatrix();
}

glm::vec3 Camera::ndcToWorld(glm::vec3 ndc) {
    glm::vec4 world = _invViewProjMatrix * glm::vec4(ndc, 1.0f);
    return glm::vec3(world) / world.w;
}

WorldFrustumPos Camera::getFrustumPos() {
    // Convert the 8 NDC frustum corners into world-space corners.
    WorldFrustumPos frustum;
    frustum.nearLB = ndcToWorld(frustum.nearLB);
    frustum.nearLT = ndcToWorld(frustum.nearLT);
    frustum.nearRB = ndcToWorld(frustum.nearRB);
    frustum.nearRT = ndcToWorld(frustum.nearRT);
    frustum.farLB = ndcToWorld(frustum.farLB);
    frustum.farLT = ndcToWorld(frustum.farLT);
    frustum.farRB = ndcToWorld(frustum.farRB);
    frustum.farRT = ndcToWorld(frustum.farRT);
    return frustum;
}

WorldFrustumPos Camera::getFrustumPos(float nearPlane, float farPlane) {
    nearPlane = std::max(0.001f, nearPlane);
    farPlane = std::max(nearPlane + 0.001f, farPlane);

    float aspect = (_screenHeight == 0)
                       ? 1.0f
                       : static_cast<float>(_screenWidth) /
                             static_cast<float>(_screenHeight);
    glm::mat4 clippedProj =
        glm::perspective(glm::radians(_FoV), aspect, nearPlane, farPlane);
    glm::mat4 clippedInvViewProj =
        glm::inverse(clippedProj * _viewMatrix);

    auto toWorld = [&](const glm::vec3& ndc) {
        glm::vec4 world = clippedInvViewProj * glm::vec4(ndc, 1.0f);
        return glm::vec3(world) / world.w;
    };

    WorldFrustumPos frustum;
    frustum.nearLB = toWorld(frustum.nearLB);
    frustum.nearLT = toWorld(frustum.nearLT);
    frustum.nearRB = toWorld(frustum.nearRB);
    frustum.nearRT = toWorld(frustum.nearRT);
    frustum.farLB = toWorld(frustum.farLB);
    frustum.farLT = toWorld(frustum.farLT);
    frustum.farRB = toWorld(frustum.farRB);
    frustum.farRT = toWorld(frustum.farRT);
    return frustum;
}

} // namespace KE
