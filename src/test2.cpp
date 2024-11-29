
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
    
    void setUp() override
    {
        //wireframe 모드
        //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        All asd = Prim::createSquareData(1.0);
        std::cout << asd.vertexAttrib.size() << std::endl;
        GLuint ss = addShape(std::make_unique<All>(asd));
        std::cout << ss << std::endl;
        std::cout << asd.indices.size() << std::endl;
        

        float vertices[] = {
            -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
            0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
            0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
            0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
            -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
            -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

            -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
            0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
            0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
            0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
            -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
            -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

            -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
            -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
            -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
            -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
            -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
            -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

            0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
            0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
            0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
            0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
            0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
            0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

            -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
            0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
            0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
            0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
            -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
            -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

            -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
            0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
            0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
            0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
            -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
            -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
        };

        unsigned int indices[] = {
            //면 1
            0, 1, 2, // 삼1
            3, 4, 5, // 삼2
            //면 2
            6, 7, 8, // 삼1
            9, 10,11, // 삼2
            //면 3
            12, 13, 14, // 삼1
            15, 16, 17, // 삼2
            //면 4
            18, 19, 20, // 삼1
            21, 22, 23, // 삼2
            //면 5
            24, 25, 26, // 삼1
            27, 28, 29, // 삼2
            //면 6
            30, 31, 32, // 삼1
            33, 34, 35 // 삼2
        };
        
        lightVao.bind();
        lightVbo.bind();
        //lightVbo.setData(sizeof(asd.vertexAttrib), &asd.vertexAttrib[0], GL_STATIC_DRAW);
        lightVbo.setData(sizeof(vertices), vertices, GL_STATIC_DRAW);
        lightEbo.bind(); // vbo는 공유해도 되지만, ebo는 각 vao마다 만들어야함.
        //lightEbo.setData(sizeof(asd.indices), &asd.indices[0], GL_STATIC_DRAW);
        lightEbo.setData(sizeof(indices), indices, GL_STATIC_DRAW);
        
        //lightVao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(asd.vertexAttrib), (void*)0);
        //lightVao.setVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(asd.vertexAttrib), (void*)offsetof(VertexAttrib, normal));
        //lightVao.setVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(asd.vertexAttrib), (void*)offsetof(VertexAttrib, uv));
        lightVao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
        lightVao.setVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3 * sizeof(float)));
        lightVao.unBind();//VAO::vaoUnBind();

        cubeShader.use();
        cubeShader.setFloat("size", size);
        cubeShader.setColor("color", color[0], color[1], color[2], color[3]);
        cubeShader.setInt("texture1", 0);
        cubeShader.setInt("texture2", 1);
        cubeShader.setVec3("lightColor",  1.0f, 1.0f, 1.0f);

        lightShader.use();
        lightShader.setVec3("objectColor", 1.0f, 0.5f, 0.31f);
        
        checkError();
    }

    void preRender() override
    {   
        std::cout << "01" << std::endl;
        checkError();
        cubeShader.use();
        glm::mat4 view = _camera.getViewMatrix();
        //_camera.updateProjMatrix(_width, _height);
        glm::mat4 projection = _camera.getProjMatrix();
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0, -1, 0));
        model = glm::scale(model, glm::vec3(0.7f));
        std::cout << "1" << std::endl;
        checkError();
        cubeShader.setMat4("model", model);
        cubeShader.setMat4("view", view);
        cubeShader.setMat4("projection", projection);
        cubeShader.setColor("color", color[0], color[1], color[2], color[3]);
        std::cout << "2" << std::endl;
        checkError();
        texture->bind(GL_TEXTURE0);
        texture2->bind(GL_TEXTURE1);
        std::cout << "3" << std::endl;
        checkError();
    }

    void render() override
    {
        /*
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
        */
        
        std::cout << "S Render---------------------" << std::endl;
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
        checkError();

        ImGui::Begin("Custom New Panel");
        ImGui::Text("You can create a panel in main loop.");
        ImGui::Checkbox("Draw Triangle", &drawTriangle);
        ImGui::SliderFloat("Size", &size, 0.5f, 2.0f);
        ImGui::ColorEdit4("Color", color);
        ImGui::End();
        checkError();
        
        cubeShader.use();
        cubeShader.setFloat("size", size);
        cubeShader.setColor("color", color[0], color[1], color[2], color[3]);
        checkError();
        std::cout << "---------------------" << std::endl;
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