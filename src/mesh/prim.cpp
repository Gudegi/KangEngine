#include "prim.hpp"

static std::vector<VertexAttrib> toVertexArrtibData(std::vector<glm::vec3>& positions, std::vector<glm::vec3>& normals, std::vector<glm::vec2>& uvs)
{
    int vnum = positions.size();
    std::vector<VertexAttrib> varray(vnum);
    for(int i = 0; i < vnum; ++i)
    {
        varray.at(i).position = positions.at(i);
        varray.at(i).normal = normals.at(i);
        varray.at(i).uv = uvs.at(i);
    }
    return varray;
}

static All createSquareData(float scale)
{
    // Scale means the length of one side.
    float half = scale / 2;
    //    v3----- v7  
    //   /|      /|   
    //  v2------v6| 
    //  | |     | |  
    //  | v0----|-v4  
    //  |/      |/
    //  v1------v5 
    //      
    std::vector<glm::vec3> positions = 
    {
        // v0, v1, v2, v3
        glm::vec3(-half, -half, -half), glm::vec3(-half, -half, half), glm::vec3(-half, half, half), glm::vec3(-half, half, -half),
        // v4, v5, v6, v7
        glm::vec3(half, -half, -half), glm::vec3(half, -half, half), glm::vec3(half, half, half), glm::vec3(half, half, -half),
        // v0, v1, v5, v4
        glm::vec3(-half, -half, -half), glm::vec3(-half, -half, half), glm::vec3(half, -half, half), glm::vec3(half, -half, -half),
        // v3, v2, v6, v7
        glm::vec3(-half, half, -half), glm::vec3(-half, half, half), glm::vec3(half, half, half), glm::vec3(half, half, -half),
        // v0, v3, v7, v4
        glm::vec3(-half, -half, -half), glm::vec3(-half, half, -half), glm::vec3(half, half, -half), glm::vec3(half, -half, -half),
        // v1, v2, v6, v5
        glm::vec3(-half, -half, half), glm::vec3(-half, half, half), glm::vec3(half, half, half), glm::vec3(half, -half, half),
    };

    std::vector<glm::vec3> normals = 
    {
        glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0),
        glm::vec3(1, 0, 0), glm::vec3(1, 0, 0), glm::vec3(1, 0, 0), glm::vec3(1, 0, 0),
        glm::vec3(0, -1, 0), glm::vec3(0, -1, 0), glm::vec3(0, -1, 0), glm::vec3(0, -1, 0),
        glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0), glm::vec3(0, 1, 0),
        glm::vec3(0, 0, -1), glm::vec3(0, 0, -1), glm::vec3(0, 0, -1), glm::vec3(0, 0, -1),
        glm::vec3(0, 0, 1), glm::vec3(0, 0, 1), glm::vec3(0, 0, 1), glm::vec3(0, 0, 1),
    };

    std::vector<glm::vec2> uvs = 
    {
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
    };

    std::vector<unsigned int> indices = {
        0, 1, 2, 0, 2, 3, // left 
        4, 6, 5, 4, 7, 6, // right
        8, 10, 9, 8, 11, 10, // down, v0,v5,v1, v0,v4,v5
        12, 13, 14, 12, 14, 15, // up
        16, 17, 18, 16, 18, 19, // back
        20, 22, 21, 20, 23, 22, // front
    };

    std::vector<VertexAttrib> attrib = toVertexArrtibData(positions, normals, uvs);
    All all;
    all.vertexAttrib = attrib;
    all.indices = indices;

    return all;
}