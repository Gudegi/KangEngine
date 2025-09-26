///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _VAO_HPP_
#define _VAO_HPP_

#include <glad/glad.h>

namespace KE {

class VAO
{

private:

    GLuint _vao;

public:

    VAO()
    {
        glGenVertexArrays(1, &_vao);
    }

    ~VAO()
    {
        glDeleteVertexArrays(1, &_vao);
    };

    void bind()
    {
        glBindVertexArray(_vao);
    }

    void unBind()
    {
        glBindVertexArray(0);
    }

    static void vaoUnBind()
    {
        glBindVertexArray(0);
    }

    void setVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer)
    { 
        glVertexAttribPointer(index, size, type, normalized, stride, pointer);
        glEnableVertexAttribArray(index);
    }

};

} // namespace KE

#endif