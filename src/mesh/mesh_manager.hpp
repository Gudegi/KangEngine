#ifndef _MESH_MANAGER_HPP_
#define _MESH_MANAGER_HPP_

#include "prim.hpp"
#include "buffer.hpp"
#include "vao.hpp"
#include "glad/glad.h"
#include <list>

// 메시 등록, 삭제, 재설정 등 관리 역할

/*
struct ShapeGlBuffer
{
    VAO vao;
    VBO vbo;
    EBO ebo;
    int numTri;
};

class MeshManager
{
private:
    //std::vector<All> _shapeList; // contents of registed shape
    std::list<std::unique_ptr<All>> _shapeLists;
    // _bufferLists // container for called in rendering loop;
    std::list<std::unique_ptr<ShapeGlBuffer>> _bufferLists;


public:
    MeshManager();
    ~MeshManager();
    void init();
    //GLuint addShape(All* infos); //return shape idx that means registed shape's index.
    GLuint addShape(std::unique_ptr<All> infos); //return shape idx that means registed shape's index.
};

*/
#endif