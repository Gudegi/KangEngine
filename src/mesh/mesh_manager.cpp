#include "mesh_manager.hpp"

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

    VAO vao;
    vao.bind();
    VBO vbo;
    vbo.bind();
    EBO ebo;
    ebo.bind();

    //vbo.setData(sizeof(vertices), vertices, GL_STATIC_DRAW);
    vbo.setData(sizeof(infos->vertexAttrib), &infos->vertexAttrib[0], GL_STATIC_DRAW);
    ebo.setData(sizeof(infos->indices), &infos->indices[0], GL_STATIC_DRAW);
    //ebo.setData(sizeof(indices), indices, GL_STATIC_DRAW);
    vao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(infos->vertexAttrib), (void*)0);
    // TODO: fixme 
    vao.setVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(infos->vertexAttrib), (void*)offsetof((infos->vertexAttrib), normal);
    vao.setVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(infos->vertexAttrib), (void*)offsetof(infos->vertexAttrib, uv));
    //vao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    //vao.setVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3 * sizeof(float)));
    vao.unBind();//VAO::vaoUnBind();

    std::unique_ptr<ShapeGlBuffer> bufferInfos;
    bufferInfos->vao = vao;
    bufferInfos->vbo = vbo;
    bufferInfos->ebo = ebo;
    bufferInfos->numTri = infos->indices.size();
    
    _bufferLists.push_back(std::move(bufferInfos));
    //_bufferLists.push_back(std::move(bufferInfos));
    //self._shape_gl_buffers[shape] = (vao, vbo, ebo, len(indices)
    return ;

}
