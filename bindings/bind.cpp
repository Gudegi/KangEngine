#include "kangEngine.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}
class PyApp: public App
{
public:
    /* Inherit the constructors */
    using App::App;

	void setup() override {
		PYBIND11_OVERRIDE_PURE(
            void, /* Return type */
            App,      /* Parent class */
            setup,          /* Name of function in C++ (must match Python name) */
                  /* Argument(s) */
        );
	}
	void preRender() override { PYBIND11_OVERRIDE_PURE(void, App, preRender, );}
	void render() override { PYBIND11_OVERRIDE_PURE(void, App, render, );}
	void postRender() override { PYBIND11_OVERRIDE_PURE(void, App, postRender, );}
};

class MyApp: public App
{
public:
    std::string defaultPath = "./build/assets/";
    Shader cubeShader = Shader(defaultPath+"shaders/test.vs", defaultPath+"shaders/test.fs");
    Shader lightShader = Shader(defaultPath+"shaders/light.vs", defaultPath+"shaders/light.fs");
    
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

        lightShader.use();
        lightShader.setVec3("objectColor", 1.0f, 0.5f, 0.31f);
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
        cubeShader.setFloat("size", size);
        cubeShader.setColor("color", color[0], color[1], color[2], color[3]);
        checkError();
        std::cout << "render End" << std::endl;
    }

    void postRender() override
    {

    }
};

class PyMyApp: public MyApp
{
public:
	using MyApp::MyApp;
	void setup() override { PYBIND11_OVERRIDE(void, MyApp, setup, );}
	void preRender() override { PYBIND11_OVERRIDE(void, MyApp, preRender, );}
	void render() override { PYBIND11_OVERRIDE(void, MyApp, render, );}
	void postRender() override { PYBIND11_OVERRIDE(void, MyApp, postRender, );}
    
};

PYBIND11_MODULE(kangengine, m) {
	// optional module docstring
    m.doc() = "pybind11 sample CashFlow module."; 
	// Add function to module.
    m.def("add", &add, "A function which adds two numbers");
    /*
    auto pyKangengine = py::class_<App>(
        m,
        "App",
        R"pbdoc(Class for performing cash flow computations.)pbdoc");
    
    pyKangengine
    .def(py::init<>())
    .def("initialize",  &App::initialize)
    .def("start",       &App::start)
    .def("setup",		&App::setup)
	.def("preRender",	&App::preRender)
	.def("render",		&App::render)
	.def("postRender",	&App::postRender);
	*/

	py::class_<App, PyApp /* <--- trampoline*/>(m, "App")
        .def(py::init<>())
		.def("initialize",  &App::initialize)
    	.def("start",       &App::start)
        .def("setup",		&App::setup)
		.def("preRender",	&App::preRender)
		.def("render",		&App::render)
		.def("postRender",	&App::postRender);

	auto pyKangengine = py::class_<MyApp, PyMyApp>(
        m,
        "MyApp",
        R"pbdoc(Class for performing cash flow computations.)pbdoc");
    
    pyKangengine
    .def(py::init<>())
    .def("initialize",  &App::initialize)
    .def("start",       &App::start);

}
