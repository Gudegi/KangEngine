#ifndef _PRIM_HPP_
#define _PRIM_HPP_

#include <glm/glm.hpp>
#include <vector>

struct VertexAttrib
{
    // A mesh consists of these datas.
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct All
{
    std::vector<VertexAttrib> vertexAttrib;
    std::vector<unsigned int> indices;
};

class Prim
{

private:


public:
    Prim();
    ~Prim();
    static std::vector<VertexAttrib> toVertexArrtibData(std::vector<glm::vec3>& positions, std::vector<glm::vec3>& normals, std::vector<glm::vec2>& uvs);
    static All createSquareData(float scale);
    static All createSphereData(float radius);


};


#endif