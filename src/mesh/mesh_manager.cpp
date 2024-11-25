#include "mesh_manager.hpp"
#include "buffer.hpp"
#include "vao.hpp"

MeshManager::MeshManager(/* args */)
{
}

MeshManager::~MeshManager()
{
}

void MeshManager::init()
{
}

GLuint MeshManager::addShape(std::unique_ptr<All> infos)
{
    _shapeLists.push_back(std::move(infos));

    VAO vao = VAO();
    vao.bind();
    Buffer vbo = Buffer(GL_ARRAY_BUFFER);
    vbo.bind();
    Buffer ebo = Buffer(GL_ELEMENT_ARRAY_BUFFER);
    ebo.bind();

    // TODO: fixme 
    //vbo.setData(sizeof(vertices), vertices, GL_STATIC_DRAW);
    //ebo.setData(sizeof(indices), indices, GL_STATIC_DRAW);
    //vao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    //vao.setVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3 * sizeof(float)));
    vao.unBind();//VAO::vaoUnBind();

}
