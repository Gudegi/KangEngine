///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _CAMERA_HPP_
#define _CAMERA_HPP_

#include <glm/glm.hpp>
#include <vector>

class Camera
{

private:
    glm::vec3 _cameraPos, _targetPos, _cameraLookDir, _cameraUpDir, _cameraRightDir;
    glm::mat4 _viewMatrix;
    glm::vec3 _upAxis;
    float _FoV, _theta, _phi;
    void updateViewMatrix();

public:
    Camera(glm::vec3 cameraPos, glm::vec3 targetPos, char upAxis='y');
    ~Camera();
    glm::vec3 getCameraPos();
    glm::vec3 getTargetPos();
    glm::mat4 getViewMatrix();
    glm::mat4 getProjMatrix(const unsigned int scrWidth, const unsigned int scrHeight, const float zNear, const float zFar);
    void setCameraPos(glm::vec3 cameraPos);
    void setTargetPos(glm::vec3 targetPos);
    void setFoV(float Fov);
    float getFoV();
    void setTheta(float theta){_theta = theta;}
    float getTheta(){return _theta;}
    void setPhi(float phi){ _phi = phi;}
    float getPhi(){return _phi;}
    //std::vector<glm::vec3> 
    glm::vec3 calcSpherePos();
    glm::vec3 getCameraLookDir();
    glm::vec3 getCameraUpDir();
    glm::vec3 getCameraRightDir();

};

#endif