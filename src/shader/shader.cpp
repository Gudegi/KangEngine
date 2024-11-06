#include "shader.hpp"

Shader::Shader(std::string vsPath, std::string fsPath)
{
    _vertexShader = compile(vsPath, GL_VERTEX_SHADER);
    _fragmentShader = compile(fsPath, GL_FRAGMENT_SHADER);
    _shaderProgram = link();
}

Shader::~Shader()
{
    glDeleteProgram(_shaderProgram);
}

std::string Shader::loadFile(const std::string path)
{   
    std::string code;
    std::ifstream file;
    file.exceptions (std::ifstream::failbit | std::ifstream::badbit);
    try 
    {
        file.open(path);
        std::stringstream shaderStream;
        shaderStream << file.rdbuf();
        file.close();
        code = shaderStream.str();
    }
    catch(std::ifstream::failure e)
    {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
    }
    
    return code;
}

GLuint Shader::compile(const std::string path, GLenum type)
{
    std::string code = loadFile(path);
    const char* p_code = code.c_str();

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &p_code, NULL);
    glCompileShader(shader);
    checkCompileError(shader);
    return shader;
}

GLuint Shader::link()
{
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, _vertexShader);
    glAttachShader(shaderProgram, _fragmentShader);
    glLinkProgram(shaderProgram);
    checkLinkError(shaderProgram);
    glDeleteShader(_vertexShader);
    glDeleteShader(_fragmentShader);
    return shaderProgram;
}

void Shader::checkCompileError(GLuint shader)
{
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success){
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
}

void Shader::checkLinkError(GLuint shaderProgram)
{
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success){
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

}

void Shader::use() const
{
    glUseProgram(_shaderProgram);
}

void Shader::setBool(const std::string &name, bool value) const
{         
    glUniform1i(glGetUniformLocation(_shaderProgram, name.c_str()), (int)value); 
}

void Shader::setInt(const std::string &name, int value) const
{ 
    glUniform1i(glGetUniformLocation(_shaderProgram, name.c_str()), value); 
}

void Shader::setFloat(const std::string &name, float value) const
{ 
    glUniform1f(glGetUniformLocation(_shaderProgram, name.c_str()), value); 
}

void Shader::setVec3(const std::string name, float x, float y, float z) const
{
    glUniform3f(glGetUniformLocation(_shaderProgram, name.c_str()), x, y, z);
}

void Shader::setColor(const std::string name, float r, float g, float b, float a) const
{
    glUniform4f(glGetUniformLocation(_shaderProgram, name.c_str()), r, g, b, a);
}
