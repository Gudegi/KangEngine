//#include "imgui.h"
//#include "imgui_impl_glfw.h"
//#include "imgui_impl_opengl3.h"
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "shader/shader.hpp"
#include "ui/panel_manager.hpp"
#include "ui/base_panel.hpp"
#include <memory>
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

    std::string default_path = "./build/assets/shaders/";
    Shader shader_program = Shader(default_path+"test.vs", default_path+"test.fs");
    PanelManager main_panel = PanelManager(window);
    BasePanel base_panel = BasePanel();
    main_panel.addPanel(&base_panel); // TODO: preventing memory leak 
    //main_panel.addPanel(std::make_unique<BasePanel>());

    // 정점 데이터( vertex data) 세팅
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };

    unsigned int VBO, VAO; //VBO를 통해 메모리를 관리한다. VAO를 통해 효율적인 렌더링을 수행한다.
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO); //VBO에게 unique 한 ID 부여
    
    //VAO 바인드 후 vertex buffer 를 세팅하고, vertex attribute 를 구성한다.
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO); //생성된 버퍼를 GL_ARRAY_BUFFER 유형으로 바인드
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); //초기화 한 정점들을 버퍼의 메모리에 복사(메모리 할당, 데이터 저장)
    
    //vertex attribute location 이 0, vec3 타입이니까 3, 정규화 NO, stride 는 float(4byte) 3개, 12도 가능, tightly packed 되어있다면 0으로도 가능, offset 은 0 void* 형으로 형변환.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3* sizeof(float), (void*)0);
    //vertex attribute 사용할 수 있게
    glEnableVertexAttribArray(0);
    
    // glVertexAttribPointer 함수로 VBO 를 vertex attribute 의 바인드 된 VBO로 등록한 후, 안전하게 언바인드 할 수 있다.
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    //glBindVertexArray(0);
    
    //wireframe 모드
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    //ImGui를 통해 조절할 변수
    bool drawTriangle = true;
    float size = 1.0f;
    float color[4] = {0.8f, 0.3f, 0.02f, 1.0f };
    
    // 변수를 쉐이더에 export
    shader_program.use();
    shader_program.setFloat("size", size);
    shader_program.setColor("color", color[0], color[1], color[2], color[3]);
    
    //렌더링 루프
    while(!glfwWindowShouldClose(window)){
        processInput(window);
        
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        main_panel.preRender();
        
        main_panel.render();
        //glBindVertexArray(VAO);
        
        if (drawTriangle)
            glDrawArrays(GL_TRIANGLES, 0, 3);

        ImGui::Begin("Custom New Panel");
        ImGui::Text("You can create a panel in main loop.");
        ImGui::Checkbox("Draw Triangle", &drawTriangle);
        ImGui::SliderFloat("Size", &size, 0.5f, 2.0f);
        ImGui::ColorEdit4("Color", color);
        ImGui::End();
        
        shader_program.setFloat("size", size);
        shader_program.setColor("color", color[0], color[1], color[2], color[3]);
        
        main_panel.postRender();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    
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