#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "kangEngine.hpp"
//#include <memory>
using namespace std;
#include <iomanip>      // std::setprecision

// TODO : fixme all

glm::vec3 scaleVec3(glm::vec3 vec3, const float s);

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

class MyApp : public App
{
public:
    void setUp() override
    {
        //wireframe 모드
        //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        std::string defaultPath = "./build/assets/";
        Shader shaderProgram = Shader(defaultPath+"shaders/test.vs", defaultPath+"shaders/test.fs");
        PanelManager mainPanel = PanelManager(app.getWindow());
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
    }

    void preRender() override
    {
        shaderProgram.use();
        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);
        view = camera.getViewMatrix();
        projection = camera.getProjMatrix(SCR_WIDTH, SCR_HEIGHT, 0.1f, 100.0f);
        shaderProgram.setMat4("view", view);
        shaderProgram.setMat4("projection", projection);
        texture.bind(GL_TEXTURE0);
        texture2.bind(GL_TEXTURE1);
        //glBindVertexArray(VAO);
        vao.bind();
    }
}

int main(){
    MyApp app = MyApp(SCR_WIDTH, SCR_HEIGHT, false);    

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
    
        shaderProgram.use();
        glm::mat4 view = glm::mat4(1.0f);
        glm::mat4 projection = glm::mat4(1.0f);
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

/*
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
*/
glm::vec3 scaleVec3(glm::vec3 vec3, const float s)
{
    vec3.x *= s;
    vec3.y *= s;
    vec3.z *= s;
    return vec3;
}
