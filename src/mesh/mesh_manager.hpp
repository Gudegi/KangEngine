#ifndef _MESH_MANAGER_HPP_
#define _MESH_MANAGER_HPP_

#include "prim.hpp"
#include "glad/glad.h"
#include <list>

// 메시 등록, 삭제, 재설정 등 관리 역할

class MeshManager
{
private:
    //std::vector<All> _shapeList; // contents of registed shape
    std::list<std::unique_ptr<All>> _shapeLists;


public:
    MeshManager(/* args */);
    ~MeshManager();
    void init();
    //GLuint addShape(All* infos); //return shape idx that means registed shape's index.
    GLuint addShape(std::unique_ptr<All> infos); //return shape idx that means registed shape's index.
};


#endif