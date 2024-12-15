/*
#include "kangEngine.hpp"
#include <iostream>
#include <memory>

const unsigned int SCR_WIDTH = 3840;
const unsigned int SCR_HEIGHT = 2160;

class MyApp: public App
{
public:
    std::string defaultPath = "./build/assets/";
    Shader cubeShader = Shader(defaultPath+"shaders/test.vs", defaultPath+"shaders/test.fs");
    Shader lightShader = Shader(defaultPath+"shaders/light.vs", defaultPath+"shaders/light.fs");
    
    VAO lightVao;
    VBO lightVbo;
    EBO lightEbo;
    Texture* texture = new Texture(defaultPath+"textures/crowdEditing.tga", false, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
    Texture* texture2 = new Texture(defaultPath+"textures/awesomeface.png", true, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

    bool drawTriangle = true;
    float size = 1.0f;
    float color[4] = {0.8f, 0.3f, 0.02f, 1.0f };    
    
    void setup() override
    {
        std::cout << "setUp called" << std::endl;
        All asd = Prim::createSquareData(1.0);

        cubeShader.use();
        cubeShader.setFloat("size", size);
        cubeShader.setColor("color", color[0], color[1], color[2], color[3]);
        cubeShader.setInt("texture1", 0);
        cubeShader.setInt("texture2", 1);
        cubeShader.setVec3("lightColor",  1.0f, 1.0f, 1.0f);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0, -1, 0));
        model = glm::scale(model, glm::vec3(0.7f));
        cubeShader.setMat4("model", model);

        GLuint ss = addShape(&cubeShader, std::make_unique<All>(asd));
        
        lightVao.bind();
        lightVbo.bind();
        lightVbo.setData(sizeof(VertexAttrib) * asd.vertexAttrib.size(), &asd.vertexAttrib[0], GL_STATIC_DRAW);
        lightEbo.bind(); // vbo는 공유해도 되지만, ebo는 각 vao마다 만들어야함.
        lightEbo.setData(sizeof(unsigned int) * asd.indices.size(), &asd.indices[0], GL_STATIC_DRAW);
        
        lightVao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexAttrib), (void*)0);
        lightVao.setVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexAttrib), (void*)offsetof(VertexAttrib, normal));
        lightVao.setVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexAttrib), (void*)offsetof(VertexAttrib, uv));
    
        lightVao.unBind();//VAO::vaoUnBind();

        lightShader.use();
        lightShader.setVec3("objectColor", 1.0f, 0.5f, 0.31f);
        
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
        //std::cout << "S Render---------------------" << std::endl;
        glm::mat4 view = _camera.getViewMatrix();
        _camera.updateProjMatrix(_width, _height);
        glm::mat4 projection = _camera.getProjMatrix();
        lightVao.bind();
        lightShader.use();
        lightShader.setMat4("view", view);
        lightShader.setMat4("projection", projection);
        glm::mat4 model = glm::mat4(1.0f);
        //model = glm::translate(model, glm::vec3(0, -2, 0));
        //model = glm::scale(model, glm::vec3(1.0f));
        lightShader.setMat4("model", model);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        lightVao.unBind();

        ImGui::Begin("Custom New Panel");
        ImGui::Text("You can create a panel in main loop.");
        ImGui::Checkbox("Draw Triangle", &drawTriangle);
        ImGui::SliderFloat("Size", &size, 0.5f, 2.0f);
        ImGui::ColorEdit4("Color", color);
        ImGui::End();
        
        cubeShader.use();
        cubeShader.setFloat("size", size);
        cubeShader.setColor("color", color[0], color[1], color[2], color[3]);
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
*/