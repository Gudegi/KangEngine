///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _BUFFER_HPP_
#define _BUFFER_HPP_

#include <glad/glad.h>

class Buffer 
{

///
/// @brief Buffer for VBO or EBO
///

private:
    GLuint _buffer;
    GLenum _target;

public:
    Buffer(GLenum target): _target(target)
    {
        glGenBuffers(1, &_buffer);
        this->bind();
    }

    ~Buffer()
    {
        glDeleteBuffers(1, &_buffer);
    }

    void bind()
    {
        glBindBuffer(_target, _buffer);
    }

    void unBind()
    {
        glBindBuffer(_target, 0);
    }   

    void setData(GLsizeiptr size, const void* data, GLenum usage) const
    {
        glBufferData(_target, size, data, usage);
    }

};

#endif