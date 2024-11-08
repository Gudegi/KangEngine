//#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "texture.hpp"
#include <iostream>

void Texture::_loadImage(const std::string path, bool flip=false)
{
    stbi_set_flip_vertically_on_load(flip);
    _data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
    if (!_data)
    {
        std::cout << "Failed to load texture" << std::endl;
        stbi_image_free(_data);
    }
}

Texture::Texture(const std::string path, bool flip): imgPath(path)
{
    this->_loadImage(path, flip);
    glGenTextures(1, &_textureID);
    this->bind();
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
    GLenum format;
    if (this->nrChannels == 1)
        format = GL_RED;
    else if (this->nrChannels == 3)
        format = GL_RGB;
    else if (this->nrChannels == 4)
        format = GL_RGBA;

    glTexImage2D(_target, 0, format, this->width, this->height, 0, format, GL_UNSIGNED_BYTE, _data);
    glGenerateMipmap(_target);
    stbi_image_free(_data);
}