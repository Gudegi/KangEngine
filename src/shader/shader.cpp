#include "shader.hpp"

Shader::Shader(std::string vsPath, std::string fsPath)
{
    _vertexShader = compile(vsPath, GL_VERTEX_SHADER);
    _fragmentShader = compile(fsPath, GL_FRAGMENT_SHADER);
    _shaderProgram = link();
}

Shader::Shader(const char* vsSource, const char* fsSource)
{
    _vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(_vertexShader, 1, &vsSource, NULL);
    glCompileShader(_vertexShader);
    checkCompileError(_vertexShader);

    _fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(_fragmentShader, 1, &fsSource, NULL);
    glCompileShader(_fragmentShader);
    checkCompileError(_fragmentShader);
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

void Shader::setColor(const std::string &name, float r, float g, float b, float a) const
{
    glUniform4f(glGetUniformLocation(_shaderProgram, name.c_str()), r, g, b, a);
}

void Shader::setVec2(const std::string &name, const glm::vec2 &value) const
{ 
    glUniform2fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, &value[0]); 
}
void Shader::setVec2(const std::string &name, float x, float y) const
{ 
    glUniform2f(glGetUniformLocation(_shaderProgram, name.c_str()), x, y); 
}

void Shader::setVec3(const std::string &name, const glm::vec3 &value) const
{ 
    glUniform3fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, &value[0]); 
}
void Shader::setVec3(const std::string &name, float x, float y, float z) const
{ 
    glUniform3f(glGetUniformLocation(_shaderProgram, name.c_str()), x, y, z); 
}

void Shader::setVec4(const std::string &name, const glm::vec4 &value) const
{ 
    glUniform4fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, &value[0]); 
}
void Shader::setVec4(const std::string &name, float x, float y, float z, float w) const
{ 
    glUniform4f(glGetUniformLocation(_shaderProgram, name.c_str()), x, y, z, w); 
}

void Shader::setMat2(const std::string &name, const glm::mat2 &mat) const
{
    glUniformMatrix2fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void Shader::setMat3(const std::string &name, const glm::mat3 &mat) const
{
    glUniformMatrix3fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void Shader::setMat4(const std::string &name, const glm::mat4 &mat) const
{
    glUniformMatrix4fv(glGetUniformLocation(_shaderProgram, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}