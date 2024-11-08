#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "shader/shader.hpp"
#include "ui/panel_manager.hpp"
#include "ui/base_panel.hpp"
#include "mesh/vao.hpp"
#include "mesh/buffer.hpp"
#include "texture/texture.hpp"
//#include <memory>
using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// 창 크기 설정
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

GLFWwindow* initGlfw(){
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "KangEngine", nullptr, nullptr);
    if(window ==NULL){
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return NULL;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    return window;
}

void initGlad(){
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        cout << "Failed to initialize GLAD" << endl;
    }
}

int main(){
    GLFWwindow* window = initGlfw();
    initGlad();

    std::string defaultPath = "./build/assets/";
    Shader shaderProgram = Shader(defaultPath+"shaders/test.vs", defaultPath+"shaders/test.fs");
    PanelManager mainPanel = PanelManager(window);
    BasePanel basePanel = BasePanel();
    mainPanel.addPanel(&basePanel); // TODO: preventing memory leak 
    //mainPanel.addPanel(std::make_unique<BasePanel>());

    // 정점 데이터( vertex data) 세팅
    float vertices[] = {
        // positions          // colors           // texture coords
        0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // top right
        0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
        -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // bottom left
        -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f    // top left 
    };

    unsigned int indices[] = {
        0, 1, 2,
        2, 3, 0
    };

    VAO vao = VAO();
    Buffer vbo = Buffer(GL_ARRAY_BUFFER);
    Buffer ebo = Buffer(GL_ELEMENT_ARRAY_BUFFER);
    Texture texture = Texture(defaultPath+"textures/crowdEditing.tga", false);
    texture.setWarpParam();
    texture.setFilterParam();
    texture.createTexture2D();

    vbo.setData(sizeof(vertices), vertices, GL_STATIC_DRAW);
    ebo.setData(sizeof(indices), indices, GL_STATIC_DRAW);
    vao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    vao.setVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3 * sizeof(float)));
    vao.setVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(6 * sizeof(float)));
    vbo.unBind();
    vao.unBind();
    
    //wireframe 모드
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    //ImGui를 통해 조절할 변수
    bool drawTriangle = true;
    float size = 1.0f;
    float color[4] = {0.8f, 0.3f, 0.02f, 1.0f };
    
    // 변수를 쉐이더에 export
    shaderProgram.use();
    shaderProgram.setFloat("size", size);
    //shaderProgram.setColor("color", color[0], color[1], color[2], color[3]);
    //shaderProgram.setInt("texture._textureID", 0);
    
    //렌더링 루프
    while(!glfwWindowShouldClose(window)){
        processInput(window);
        
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        mainPanel.preRender();
        
        mainPanel.render();
        texture.bind(GL_TEXTURE0);
        shaderProgram.use();
        //glBindVertexArray(VAO);
        vao.bind();
        if (drawTriangle)
            //glDrawArrays(GL_TRIANGLES, 0, 3);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        vao.unBind();

        ImGui::Begin("Custom New Panel");
        ImGui::Text("You can create a panel in main loop.");
        ImGui::Checkbox("Draw Triangle", &drawTriangle);
        ImGui::SliderFloat("Size", &size, 0.5f, 2.0f);
        ImGui::ColorEdit4("Color", color);
        ImGui::End();
        
        shaderProgram.setFloat("size", size);
        //shaderProgram.setColor("color", color[0], color[1], color[2], color[3]);
        
        mainPanel.postRender();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    //glDeleteVertexArrays(1, &VAO);
    //glDeleteBuffers(1, &VBO);
    //glDeleteBuffers(1, &EBO);
    
    glfwTerminate();
    return 0;
}
    
    void framebuffer_size_callback(GLFWwindow* window, int width, int height ){
        glViewport(0, 0, width, height);
    }
    
    void processInput(GLFWwindow* window) {
        if( glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
    }