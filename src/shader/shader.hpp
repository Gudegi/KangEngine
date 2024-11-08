///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _SHADER_HPP_
#define _SHADER_HPP_

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class Shader
{

///
/// @brief  The use of one shader program by receiving the vs and fs file paths.
///

private:

    ///
    /// @brief Variables conatin shader programs and complied shaders.
    ///
    GLuint _shaderProgram, _vertexShader, _fragmentShader;
    const std::string _vsPath, _fsPath;

    ///
    /// @brief Load file from string path. This function is called in "compile" function.
    /// @param The path of the shader.
    ///
    std::string loadFile(const std::string path);

    ///
    /// @brief Compile the shader. This function is called in Shader() function.
    /// @param The path of the shader and type of shader (ig, GL_VERTEX_SHADER)
    ///
    GLuint compile(const std::string path, GLenum type); // compile the shader

    ///
    /// @brief Link the shader and shader program. This function is called in Shader() function.
    ///
    GLuint link(); // return shader program


    ///
    /// @brief Check compile error.
    ///
    void checkCompileError(GLuint shader);

    ///
    /// @brief Check link error.
    ///
    void checkLinkError(GLuint shaderProgram);

public:

    ///
    /// @brief Construct a new shader object.
    /// @param The paths of vertex and fragment shader.
    /// @todo Support another shader (ig, geometry shader).
    ///
    Shader(std::string vsPath, std::string fsPath);

    ///
    /// @brief Destruct object.
    ///
    ~Shader();

    ///
    /// @brief Activate the shader
    ///
    void use() const;

    ///
    /// @brief Change shader variables
    /// @todo Need more types.
    ///
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setColor(const std::string &name, float r, float g, float b, float a) const;

    void setVec2(const std::string &name, const glm::vec2 &value) const;
    void setVec2(const std::string &name, float x, float y) const;
    void setVec3(const std::string &name, const glm::vec3 &value) const;
    void setVec3(const std::string &name, float x, float y, float z) const;
    void setVec4(const std::string &name, const glm::vec4 &value) const;
    void setVec4(const std::string &name, float x, float y, float z, float w) const;
    void setMat2(const std::string &name, const glm::mat2 &mat) const;
    void setMat3(const std::string &name, const glm::mat3 &mat) const;
    void setMat4(const std::string &name, const glm::mat4 &mat) const;
};

#endif