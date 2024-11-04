#ifndef _SHADER_HPP_
#define _SHADER_HPP_

#include <glad/glad.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class Shader{

/*
    Define the use of one shader program by receiving the vs and fs file paths.
*/

private:
    GLuint m_shader_program, m_v_shader, m_f_shader;
    const std::string m_vs_path, m_fs_path;

    std::string loadFile(const std::string path);
    GLuint compile(const std::string path, GLenum type); // compile the shader
    GLuint link(); // return shader program
    void checkCompileError(GLuint shader);
    void checkLinkError(GLuint shader_program);

public:
    Shader(std::string vs_path, std::string fs_path);
    ~Shader();
    void use() const;
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setVec3(const std::string name, float x, float y, float z) const;
    void setColor(const std::string name, float r, float g, float b, float a) const;
};

#endif