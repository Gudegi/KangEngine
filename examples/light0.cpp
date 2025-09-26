
#include "kangEngine.hpp"
#include <iostream>
#include <memory>

using namespace KE;

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

///
/// LearnOpenGL : Lighting - Colors
///

const char* lightFs = R"(
#version 410 core

out vec4 FragColor;

in vec2 TexCoord;
uniform vec3 lightColor;

void main() {
   FragColor = vec4(lightColor, 1.0);
}
)";

const char* lightVs = R"(
#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
//layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
}
)";

const char* testFs = R"(
#version 410 core

out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1, texture2;
uniform vec4 objColor;
uniform vec3 lightColor;

void main() {
   FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2) * vec4(objColor) * vec4(lightColor, 1.0f);
}
)";

const char* testVs = R"(
#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
}
)";

class MyApp: public App
{
public:
    std::string defaultPath = "./build/assets/";
    //Shader cubeShader = Shader(defaultPath+"shaders/test.vs", defaultPath+"shaders/test.fs");
    Shader cubeShader = Shader(testVs, testFs);
    Shader lightShader = Shader(lightVs, lightFs);
    
    Texture* texture = new Texture(defaultPath+"textures/crowdEditing.tga", false, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
    Texture* texture2 = new Texture(defaultPath+"textures/awesomeface.png", true, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

    bool drawTriangle = true;
    float size = 0.7f;
    float color[4] = {0.8f, 0.3f, 0.02f, 1.0f }; 
    float lightColor[3] = {1.0f, 1.0f, 1.0f};
    
    void setup() override
    {
        std::cout << "setUp called" << std::endl;
        All asd = Prim::createSquareData(1.0);

        cubeShader.use();
        cubeShader.setFloat("size", size);
        cubeShader.setColor("objColor", color[0], color[1], color[2], color[3]);
        cubeShader.setInt("texture1", 0);
        cubeShader.setInt("texture2", 1);
        cubeShader.setVec3("lightColor",  lightColor[0], lightColor[1], lightColor[2]);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0, -1, 0));
        model = glm::scale(model, glm::vec3(size));
        cubeShader.setMat4("model", model);

        lightShader.use();
        lightShader.setVec3("lightColor",  lightColor[0], lightColor[1], lightColor[2]);
        model = glm::mat4(1.0f);
        //model = glm::translate(model, glm::vec3(0, -2, 0));
        //model = glm::scale(model, glm::vec3(1.0f));
        lightShader.setMat4("model", model);

        GLuint s1 = addShape(&cubeShader, std::make_unique<All>(asd));
        GLuint s2 = addShape(&lightShader, std::make_unique<All>(asd));
        
        checkError();
        std::cout << "setUp End" << std::endl;
    }

    void preRender() override
    {   
        std::cout << "preRender Called" << std::endl;
        cubeShader.use();
        cubeShader.setColor("color", color[0], color[1], color[2], color[3]);
        texture->bind(GL_TEXTURE0);
        texture2->bind(GL_TEXTURE1);
        checkError();
        std::cout << "preRender End" << std::endl;
    }

    void render() override
    {
        std::cout << "render Called" << std::endl;

        ImGui::Begin("Custom New Panel");
        ImGui::Text("You can create a panel in main loop.");
        ImGui::Checkbox("Draw Triangle", &drawTriangle);
        ImGui::SliderFloat("Size", &size, 0.5f, 2.0f);
        ImGui::ColorEdit4("Color", color);
        ImGui::End();
        
        cubeShader.use();
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0, -1, 0));
        model = glm::scale(model, glm::vec3(size));
        cubeShader.setMat4("model", model);
        cubeShader.setColor("objColor", color[0], color[1], color[2], color[3]);
        checkError();
        std::cout << "render End" << std::endl;
    }

    void postRender() override
    {

    }
};

int main(){
    MyApp app;
    app.initialize(SCR_WIDTH, SCR_HEIGHT, false);
    app.start();
    return 0;
}
