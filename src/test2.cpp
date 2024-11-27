
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
    
    VAO cubeVao;
    VBO cubeVbo;
    EBO cubeEbo;

    VAO lightVao = VAO();
    VBO lightEbo;
    Texture* texture = new Texture(defaultPath+"textures/crowdEditing.tga", false, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
    Texture* texture2 = new Texture(defaultPath+"textures/awesomeface.png", true, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

    bool drawTriangle = true;
    float size = 1.0f;
    float color[4] = {0.8f, 0.3f, 0.02f, 1.0f };    
    
    void setUp() override
    {
        //wireframe 모드
        //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        All asd = Prim::createSquareData(1.0);
        std::cout << asd.vertexAttrib.size() << std::endl;
        std::cout << asd.indices.size() << std::endl;
        
               
        cubeVao.bind();
        cubeVbo.bind();
        cubeEbo.bind();
        cubeVbo.setData(sizeof(vertices), vertices, GL_STATIC_DRAW);
        cubeEbo.setData(sizeof(indices), indices, GL_STATIC_DRAW);
        cubeVao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
        cubeVao.setVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3 * sizeof(float)));
        cubeVao.unBind();//VAO::vaoUnBind();

        lightVao.bind();
        lightEbo.bind(); // vbo는 공유해도 되지만, ebo는 각 vao마다 만들어야함.
        lightEbo.setData(sizeof(indices), indices, GL_STATIC_DRAW);
        lightVao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
        lightVao.setVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3 * sizeof(float)));
        lightVao.unBind();//VAO::vaoUnBind();

        Buffer::vboUnBind();

        cubeShader.use();
        cubeShader.setFloat("size", size);
        cubeShader.setColor("color", color[0], color[1], color[2], color[3]);
        cubeShader.setInt("texture1", 0);
        cubeShader.setInt("texture2", 1);
        cubeShader.setVec3("lightColor",  1.0f, 1.0f, 1.0f);

        //lightShader.use();
        //lightShader.setVec3("objectColor", 1.0f, 0.5f, 0.31f);
        
        checkError();
    }

    void preRender() override
    {   
    }

    void render() override
    {
        cubeShader.use();
        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);
        view = _camera.getViewMatrix();
        projection = _camera.getProjMatrix(_width, _height, 0.1f, 100.0f);
        cubeShader.setMat4("view", view);
        cubeShader.setMat4("projection", projection);
        cubeShader.setColor("color", color[0], color[1], color[2], color[3]);

        texture->bind(GL_TEXTURE0);
        texture2->bind(GL_TEXTURE1);
        
        cubeVao.bind();
        glm::vec3 cubePositions[] = {
            glm::vec3( 0.0f,  0.0f,  0.0f),
            glm::vec3( 2.0f,  5.0f, -15.0f),
            glm::vec3(-1.5f, -2.2f, -2.5f),
            glm::vec3(-3.8f, -2.0f, -12.3f),
            glm::vec3( 2.4f, -0.4f, -3.5f),
            glm::vec3(-1.7f,  3.0f, -7.5f),
            glm::vec3( 1.3f, -2.0f, -2.5f),
            glm::vec3( 1.5f,  2.0f, -2.5f),
            glm::vec3( 1.5f,  0.2f, -1.5f),
            glm::vec3(-1.3f,  1.0f, -1.5f)
        };
        for (int i=0; i<=9; i++)
        {   
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, cubePositions[i]);
            float angle = (i+1) * 20.0f;
            model = glm::rotate(model, (float)glfwGetTime() * glm::radians(angle), glm::vec3(0.5f, 1.0f, 0.0));
            cubeShader.setMat4("model", model);
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        }
        cubeVao.unBind();

        lightVao.bind();
        lightShader.use();
        lightShader.setMat4("view", view);
        lightShader.setMat4("projection", projection);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0, -2, 0));
        model = glm::scale(model, glm::vec3(0.7f));
        lightShader.setMat4("model", model);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        lightVao.unBind();


        ImGui::Begin("Custom New Panel");
        ImGui::Text("You can create a panel in main loop.");
        ImGui::Checkbox("Draw Triangle", &drawTriangle);
        ImGui::SliderFloat("Size", &size, 0.5f, 2.0f);
        ImGui::ColorEdit4("Color", color);
        ImGui::End();
        
        cubeShader.setFloat("size", size);
        cubeShader.setColor("color", color[0], color[1], color[2], color[3]);
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