#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "kangEngine.hpp"
#include <memory>
using namespace std;
#include <iomanip>      // std::setprecision

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window, Camera* camera);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

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
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSwapInterval(1);
    return window;
}

void initGlad(){
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        cout << "Failed to initialize GLAD" << endl;
    }
}

float deltaTime = 0.0f;
float lastFrameTime = 0.0f;

bool isMouseMiddleClicked = false;
bool isMouseLeftClicked = false;
double mouseX = 0.0;
double mouseY = 0.0;
double prevMouseX = 0.0;
double prevMouseY = 0.0;
double deltaMouseX = 0.0;
double deltaMouseY = 0.0;

glm::vec3 cameraPos = glm::vec3(3, 2, -1);
glm::vec3 cameraTarget = glm::vec3(0, 0, 0);
Camera camera = Camera(cameraPos, cameraTarget, 'y');

int main(){
    GLFWwindow* window = initGlfw();
    initGlad();
    glEnable(GL_DEPTH_TEST);

    std::string defaultPath = "./build/assets/";
    Shader shaderProgram = Shader(defaultPath+"shaders/test.vs", defaultPath+"shaders/test.fs");
    PanelManager mainPanel;
    mainPanel.init(window);
    BasePanel basePanel = BasePanel();
    mainPanel.addPanel(&basePanel); // TODO: preventing memory leak 
    //mainPanel.addPanel(std::make_unique<BasePanel>());

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
    
    // world space positions of our cubes
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
    
    VAO vao = VAO();
    Buffer vbo = Buffer(GL_ARRAY_BUFFER);
    Buffer ebo = Buffer(GL_ELEMENT_ARRAY_BUFFER);
    Texture texture = Texture(defaultPath+"textures/crowdEditing.tga", false);
    texture.setWarpParam();
    texture.setFilterParam();
    texture.createTexture2D();
    Texture texture2 = Texture(defaultPath+"textures/awesomeface.png", true);
    texture2.setWarpParam();
    texture2.setFilterParam();
    texture2.createTexture2D();

    vbo.setData(sizeof(vertices), vertices, GL_STATIC_DRAW);
    ebo.setData(sizeof(indices), indices, GL_STATIC_DRAW);
    vao.setVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    vao.setVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3 * sizeof(float)));
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
    shaderProgram.setInt("texture1", 0);
    shaderProgram.setInt("texture2", 1);

    //렌더링 루프
    while(!glfwWindowShouldClose(window)){
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrameTime;
        lastFrameTime = currentFrame;
        processInput(window, &camera);
        
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        mainPanel.preRender();
        mainPanel.render();


        shaderProgram.use();
        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);
        //model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        //view  = glm::translate(view, glm::vec3(0.0f, 0.0f, -13.0f));
        //glm::vec3 cPos = camera.getCameraPos();
        //const float radius = 10.0f;
        //float camX = sin(glfwGetTime()) * radius;
        //float camZ = cos(glfwGetTime()) * radius;
        //cPos = glm::vec3(camX, cPos.y, camZ);
        //camera.setCameraPos(cPos);
        view = camera.getViewMatrix();
        projection = camera.getProjMatrix(SCR_WIDTH, SCR_HEIGHT, 0.1f, 100.0f);
        shaderProgram.setMat4("view", view);
        shaderProgram.setMat4("projection", projection);
        texture.bind(GL_TEXTURE0);
        texture2.bind(GL_TEXTURE1);
        //glBindVertexArray(VAO);
        vao.bind();
        if (drawTriangle)
            //glDrawArrays(GL_TRIANGLES, 0, 3);
        for (int i=0; i<=9; i++)
        {   
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, cubePositions[i]);
            float angle = (i+1) * 20.0f;
            model = glm::rotate(model, (float)glfwGetTime() * glm::radians(angle), glm::vec3(0.5f, 1.0f, 0.0));
            shaderProgram.setMat4("model", model);
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        }
            
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


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    deltaMouseX = 0;
    deltaMouseY = 0;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        isMouseLeftClicked = true;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
        isMouseLeftClicked = false;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
        isMouseMiddleClicked = true;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_RELEASE)
        isMouseMiddleClicked = false;    
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) 
{
    mouseX = xpos;
    mouseY = ypos;
    if (isMouseLeftClicked == true || isMouseMiddleClicked == true)
    {
        deltaMouseX = mouseX - prevMouseX;
        deltaMouseY = mouseY - prevMouseY;
    }
    else
    {
        deltaMouseX = 0;
        deltaMouseY = 0;
    }
    prevMouseX = xpos;
    prevMouseY = ypos;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    float fov = camera.getFoV();
    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
        glm::vec3 cameraPos = camera.getCameraPos();
        glm::vec3 cameraFront = camera.getCameraLookDir();
        float cameraSpeed = static_cast<float>(10.0 * deltaTime);
        cameraPos -= cameraSpeed * cameraFront;
        camera.setCameraPos(cameraPos);
    if (fov > 60.0f)
        fov = 60.0f; 
    camera.setFoV(fov);
}
    
void processInput(GLFWwindow* window, Camera* camera) {
    float cameraSpeed = static_cast<float>(15.0 * deltaTime);
    float scaleDistance = camera->getCamToLookDistance();
    glm::vec3 cameraPos = camera->getCameraPos();
    glm::vec3 cameraFront = camera->getCameraLookDir();
    glm::vec3 cameraUp = camera->getCameraUpDir();
    glm::vec3 cameraRight = camera->getCameraRightDir();
    glm::vec3 globalUp = glm::vec3(0.0f, 1.0f, 0.0f);

    if( glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    
    // look at specifc target or // look at forward direction 
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        {
            if (isMouseLeftClicked == true)
            {   
                glm::vec3 nextCameraPos = cameraPos + cameraSpeed * cameraUp;
                glm::vec3 nextCameraFront = camera->getTargetPos() - nextCameraPos; // No normalize because it is just for sign checking.
                if (((nextCameraFront.x * cameraFront.x) < 0) && (nextCameraFront.z * cameraFront.z) < 0)
                    return; // To prevent flipping, compare nextCameraFront with cameraFront.
                cameraPos = nextCameraPos;
            }
            else
            {
                glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                camera->setTargetPos(cameraPos + scaledFront);
                //cameraPos += cameraSpeed * cameraFront;
                cameraPos += cameraSpeed * cameraUp;
                camera->setTargetPos(cameraPos + scaledFront); // prevent disconnect
            }
            camera->setCameraPos(cameraPos);
        }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        {        
            if (isMouseLeftClicked == true)
            {
                glm::vec3 nextCameraPos = cameraPos - cameraSpeed * cameraUp;
                glm::vec3 nextCameraFront = camera->getTargetPos() - nextCameraPos;
                if (((nextCameraFront.x * cameraFront.x) < 0) && (nextCameraFront.z * cameraFront.z) < 0)
                    return;
                cameraPos = nextCameraPos;
            }
            else
            {
                glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                camera->setTargetPos(cameraPos + scaledFront);
                //cameraPos -= cameraSpeed * cameraFront;
                cameraPos -= cameraSpeed * cameraUp;
                camera->setTargetPos(cameraPos + scaledFront);
            }
            camera->setCameraPos(cameraPos);
        }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        {
            if (isMouseLeftClicked == true)
            {
                cameraPos -= cameraSpeed * cameraRight;
            }
            else
            {
                glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                camera->setTargetPos(cameraPos + scaledFront);
                cameraPos -= cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
                camera->setTargetPos(cameraPos + scaledFront);
            }
            camera->setCameraPos(cameraPos);
        }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        {
            if (isMouseLeftClicked == true)
            {
                cameraPos += cameraSpeed * cameraRight;
            }
            else
            {
                glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
                camera->setTargetPos(cameraPos + scaledFront);
                cameraPos += cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
                camera->setTargetPos(cameraPos + scaledFront);
            }
            camera->setCameraPos(cameraPos);
        }

    if (isMouseLeftClicked) 
    {
        const double DRAG_THRESHOLD = 2.0;
        if (sqrt(deltaMouseX * deltaMouseX + deltaMouseY * deltaMouseY) < DRAG_THRESHOLD)
        {
            return;
        }
        if (deltaMouseX > 0) 
        {
            cameraPos -= cameraSpeed * cameraRight;
        }
        if (deltaMouseX < 0)
        { 
            cameraPos += cameraSpeed * cameraRight;
        }
        if (deltaMouseY > 0) 
        {
            glm::vec3 nextCameraPos = cameraPos - cameraSpeed * cameraUp;
            glm::vec3 nextCameraFront = camera->getTargetPos() - nextCameraPos;
            if (((nextCameraFront.x * cameraFront.x) < 0) && (nextCameraFront.z * cameraFront.z) < 0)
                return;
            cameraPos = nextCameraPos;
        }
        if (deltaMouseY < 0) 
        {
            glm::vec3 nextCameraPos = cameraPos + cameraSpeed * cameraUp;
            glm::vec3 nextCameraFront = camera->getTargetPos() - nextCameraPos; // No normalize because it is just for sign checking.
            if (((nextCameraFront.x * cameraFront.x) < 0) && (nextCameraFront.z * cameraFront.z) < 0)
                return; // To prevent flipping, compare nextCameraFront with cameraFront.
            cameraPos = nextCameraPos;
        }
        camera->setCameraPos(cameraPos);
    }

    if (isMouseMiddleClicked) 
    {
        const double DRAG_THRESHOLD = 2.0;
        if (sqrt(deltaMouseX * deltaMouseX + deltaMouseY * deltaMouseY) < DRAG_THRESHOLD)
        {
            return;
        }
        if (deltaMouseX > 0) 
        {
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            camera->setTargetPos(cameraPos + scaledFront);
            cameraPos -= cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
            camera->setTargetPos(cameraPos + scaledFront);
        }
        if (deltaMouseX < 0)
        { 
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            camera->setTargetPos(cameraPos + scaledFront);
            cameraPos += cameraSpeed * glm::normalize(glm::cross(cameraFront, globalUp));
            camera->setTargetPos(cameraPos + scaledFront);
        }
        if (deltaMouseY > 0) 
        {
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            camera->setTargetPos(cameraPos + scaledFront);
            //cameraPos += cameraSpeed * cameraFront;
            cameraPos += cameraSpeed * cameraUp;
            camera->setTargetPos(cameraPos + scaledFront);
        }
        if (deltaMouseY < 0) 
        {
            glm::vec3 scaledFront = scaleVec3(cameraFront, scaleDistance);
            camera->setTargetPos(cameraPos + scaledFront);
            //cameraPos -= cameraSpeed * cameraFront;
            cameraPos -= cameraSpeed * cameraUp;
            camera->setTargetPos(cameraPos + scaledFront);
        }
        camera->setCameraPos(cameraPos);
    }
}
