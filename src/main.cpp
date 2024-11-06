//#include "imgui.h"
//#include "imgui_impl_glfw.h"
//#include "imgui_impl_opengl3.h"
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "shader/shader.hpp"
#include "ui/panel_manager.hpp"
#include "ui/base_panel.hpp"
#include "mesh/vao.hpp"
#include "mesh/buffer.hpp"
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

    std::string defaultPath = "./build/assets/shaders/";
    Shader shaderProgram = Shader(defaultPath+"test.vs", defaultPath+"test.fs");
    PanelManager mainPanel = PanelManager(window);
    BasePanel basePanel = BasePanel();
    mainPanel.addPanel(&basePanel); // TODO: preventing memory leak 
    //mainPanel.addPanel(std::make_unique<BasePanel>());

    // 정점 데이터( vertex data) 세팅
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
        -0.5f, 0.5f, 0.0f,
         0.5f, 0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
    };

    unsigned int indices[] = {
        0, 1, 2,
        2, 3, 0
    };

    VAO vao = VAO();
    Buffer vbo = Buffer(GL_ARRAY_BUFFER);
    Buffer ebo = Buffer(GL_ELEMENT_ARRAY_BUFFER);

    vao.bind();
    vbo.bind();
    vbo.setData(sizeof(vertices), vertices, GL_STATIC_DRAW);
    ebo.bind();
    ebo.setData(sizeof(indices), indices, GL_STATIC_DRAW);
    vao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    vbo.unBind();
    vao.unBind();
    /*
    GLuint VBO, VAO, EBO; //VBO를 통해 메모리를 관리한다. VAO를 통해 효율적인 렌더링을 수행한다.
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO); //VBO에게 unique 한 ID 부여
    glGenBuffers(1, &EBO); 
    
    //VAO 바인드 후 vertex buffer 를 세팅하고, vertex attribute 를 구성한다.
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO); //생성된 버퍼를 GL_ARRAY_BUFFER 유형으로 바인드
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); //초기화 한 정점들을 버퍼의 메모리에 복사(메모리 할당, 데이터 저장)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    //vertex attribute location 이 0, vec3 타입이니까 3, 정규화 NO, stride 는 float(4byte) 3개, 12도 가능, tightly packed 되어있다면 0으로도 가능, offset 은 0 void* 형으로 형변환.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3* sizeof(float), (void*)0);
    //vertex attribute 사용할 수 있게
    glEnableVertexAttribArray(0);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    */
    
    //wireframe 모드
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    //ImGui를 통해 조절할 변수
    bool drawTriangle = true;
    float size = 1.0f;
    float color[4] = {0.8f, 0.3f, 0.02f, 1.0f };
    
    // 변수를 쉐이더에 export
    shaderProgram.use();
    shaderProgram.setFloat("size", size);
    shaderProgram.setColor("color", color[0], color[1], color[2], color[3]);
    
    //렌더링 루프
    while(!glfwWindowShouldClose(window)){
        processInput(window);
        
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        mainPanel.preRender();
        
        mainPanel.render();
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
        shaderProgram.setColor("color", color[0], color[1], color[2], color[3]);
        
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