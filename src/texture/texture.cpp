//#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "texture.hpp"
#include <iostream>

unsigned char* Texture::_loadImage(const std::string path, bool flip=false)
{
    stbi_set_flip_vertically_on_load(flip);
    unsigned char* data = stbi_load(path.c_str(), &_width, &_height, &_nChannels, 0);
    if (!data)
    {
        std::cout << "Failed to load texture" << std::endl;
        stbi_image_free(data);
    }
    return data;
}

Texture::Texture(const std::string path, bool flip, GLfloat warpParam, GLfloat minFilferParam, GLfloat maxFilterParam): imgPath(path)
{
    glGenTextures(1, &_textureID);
    unsigned char* data = this->_loadImage(path, flip);
    this->bind();
    GLenum format;
    if (this->_nChannels == 1)
        format = GL_RED;
    else if (this->_nChannels == 3)
        format = GL_RGB;
    else if (this->_nChannels == 4)
        format = GL_RGBA;

    glTexImage2D(_target, 0, format, this->_width, this->_height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(_target);

    this->setWarpParam(warpParam);
    this->setFilterParam(minFilferParam, maxFilterParam);
    stbi_image_free(data);
}

Texture::~Texture()
{
 
}

void Texture::bind() const
{
    glBindTexture(_target, _textureID);
}

void Texture::bind(GLenum texture) const
{   
    glActiveTexture(texture);
    this->bind();
}

void Texture::setWarpParam(GLfloat param) const
{
    glTexParameteri(_target, GL_TEXTURE_WRAP_S, param);
    glTexParameteri(_target, GL_TEXTURE_WRAP_T, param);
}

void Texture::setFilterParam(GLfloat minParam, GLfloat maxParam) const
{
    glTexParameteri(_target, GL_TEXTURE_MIN_FILTER, minParam);
    glTexParameteri(_target, GL_TEXTURE_MAG_FILTER, maxParam);
}


void Texture::createTexture2D()
{
    // buggy
    /*
    GLenum format;
    if (this->_nChannels == 1)
        format = GL_RED;
    else if (this->_nChannels == 3)
        format = GL_RGB;
    else if (this->_nChannels == 4)
        format = GL_RGBA;

    glTexImage2D(_target, 0, format, this->_width, this->_height, 0, format, GL_UNSIGNED_BYTE, _data);
    glGenerateMipmap(_target);
    */
}