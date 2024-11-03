#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// 창 크기 설정
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

//각 shader GLSL 코드
const char *vertexShaderSource = "#version 330 core\n"
        "layout (location = 0) in vec3 aPos;\n"
        "uniform float size;\n"
        //out vec4 vertexColor;\n" //// chap - Shader
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(size*aPos.x, size*aPos.y, size*aPos.z, 1.0);\n"
        //"vertexColor = vec4(0.5, 0.0, 0.0, 1.0);\n" //// chap - Shader
    "}\0";

const char *fragmentShaderSource = "#version 330 core\n"
        "out vec4 FragColor;\n"
        "uniform vec4 color;\n"
        //"in vec4 vertexColor;"

        "void main() {\n"
            //"FragColor = vertexColor;\n" // chap - Shader
            //"FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
            "FragColor = color;\n "
        "}\n\0";


int main(){
    
    // glfw : 초기화, 설정
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // MAC OSX
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    
    // glfw window 생성
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", nullptr, nullptr);
    if(window ==NULL){
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    // OpenGL 함수 포인터 로드
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        cout << "Failed to initialize GLAD" << endl;
        return -1;
    }
    
    
    //vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // 컴파일 에러 체크
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success){
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << endl;
    }
    //fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    //  컴파일 에러 체크
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success){
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << endl;
    }
    // link shaders
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // 링킹 에러 체크
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success){
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << endl;
    }
    //링크 했으니 shader 객체는 제거
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
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
    glBindVertexArray(0);
    
    //wireframe 모드
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    
    //ImGui 초기화
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    float xscale, yscale;
    GLFWmonitor* monitor = glfwGetWindowMonitor(window);
    if (!monitor) {
        monitor = glfwGetPrimaryMonitor();
    }
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);
    float dpi_scale = (xscale + yscale) * 0.5f;
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(dpi_scale);
    io.FontGlobalScale = dpi_scale;
    //ImFontConfig config;
    //config.SizePixels = 16.0f * dpi_scale;
    //io.Fonts->AddFontDefault(&config);
    //io.Fonts->Build();
    
    //ImGui를 통해 조절할 변수
    bool drawTriangle = true;
    float size = 1.0f;
    float color[4] = {0.8f, 0.3f, 0.02f, 1.0f };
    
    // 변수를 쉐이더에 export
    glUseProgram(shaderProgram);
    glUniform1f(glGetUniformLocation(shaderProgram, "size"), size);
    glUniform4f(glGetUniformLocation(shaderProgram, "color"), color[0], color[1], color[2], color[3]);
    
    //렌더링 루프
    while(!glfwWindowShouldClose(window)){
        
        
        processInput(window);
        
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        // OpenGL에게 ImGui시작 알림
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        
        if (!io.WantCaptureMouse){
            // Your input functions here
        }
        
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        
        //체크되어야만 그림
        if (drawTriangle)
            glDrawArrays(GL_TRIANGLES, 0, 3);
        
        //ImGui 윈도우 생성
        ImGui::Begin("Hello, world!");
        //텍스트
        ImGui::Text("This is some useful text.");
        //체크박스
        ImGui::Checkbox("Draw Triangle", &drawTriangle);
        //슬라이더
        ImGui::SliderFloat("Size", &size, 0.5f, 2.0f);
        //컬러 에디터
        ImGui::ColorEdit4("Color", color);
        //ImGui 종료
        ImGui::End();
        
        
        glUseProgram(shaderProgram);
        glUniform1f(glGetUniformLocation(shaderProgram, "size"), size);
        glUniform4f(glGetUniformLocation(shaderProgram, "color"), color[0], color[1], color[2], color[3]);
        
        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
        glfwPollEvents();
        
        
    }
    
    // 인스턴스 삭제
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    
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