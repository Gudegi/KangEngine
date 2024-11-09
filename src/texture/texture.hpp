#ifndef _TEXTURE_HPP_
#define _TEXTURE_HPP_

#include <string>
#include <glad/glad.h>

class Texture
{

private:

    GLuint _textureID;
    GLenum _target = GL_TEXTURE_2D;
    int _width, _height, _nChannels;
    unsigned char* _data;
    void _loadImage(const std::string path, bool flip);

public:

    const std::string imgPath;

    Texture(const std::string path, bool flip=false);

    ~Texture();

    void bind() const;

    ///
    ///
    /// @param ex) GL_TEXTURE0
    ///
    void bind(GLenum texture) const;
    void setWarpParam(GLfloat param=GL_REPEAT) const;
    void setFilterParam(GLfloat minParam=GL_LINEAR_MIPMAP_LINEAR, GLfloat maxParam=GL_REPEAT) const;
    void createTexture2D();
    //void createTexture3D();




};



#endif