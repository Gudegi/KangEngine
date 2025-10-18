///
/// KangEngine Python Bindings
///

#include "../src/kangEngine.hpp"
#include "../src/backend/graphics_factory.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace py = pybind11;
using namespace KE;

// Forward declaration for scene bindings
void bind_scene(py::module& m);

// Trampoline class for App - allows Python to override virtual methods
class PyApp: public App
{
public:
    using App::App;

	void setup() override {
		PYBIND11_OVERRIDE_PURE(void, App, setup);
	}
	void preRender() override {
		PYBIND11_OVERRIDE_PURE(void, App, preRender);
	}
	void render() override {
		PYBIND11_OVERRIDE_PURE(void, App, render);
	}
	void postRender() override {
		PYBIND11_OVERRIDE_PURE(void, App, postRender);
	}
};

PYBIND11_MODULE(kangengine, m) {
    m.doc() = "KangEngine Python bindings";

	// Backend type enum
	py::enum_<Backend::BackendType>(m, "BackendType")
		.value("OpenGL", Backend::BackendType::OpenGL)
		.value("Vulkan", Backend::BackendType::Vulkan)
		.value("WebGPU", Backend::BackendType::WebGPU)
		.export_values();

	// Backend::Shader
	py::class_<Backend::Shader>(m, "Shader")
		.def("use", &Backend::Shader::use)
		.def("bind", &Backend::Shader::bind)
		.def("unbind", &Backend::Shader::unbind)
		.def("setBool", &Backend::Shader::setBool)
		.def("setInt", &Backend::Shader::setInt)
		.def("setFloat", &Backend::Shader::setFloat)
		.def("setColor", &Backend::Shader::setColor)
		.def("setVec2", py::overload_cast<const std::string&, const glm::vec2&>(&Backend::Shader::setVec2))
		.def("setVec2", py::overload_cast<const std::string&, float, float>(&Backend::Shader::setVec2))
		.def("setVec3", py::overload_cast<const std::string&, const glm::vec3&>(&Backend::Shader::setVec3))
		.def("setVec3", py::overload_cast<const std::string&, float, float, float>(&Backend::Shader::setVec3))
		.def("setVec4", py::overload_cast<const std::string&, const glm::vec4&>(&Backend::Shader::setVec4))
		.def("setVec4", py::overload_cast<const std::string&, float, float, float, float>(&Backend::Shader::setVec4))
		.def("setMat2", &Backend::Shader::setMat2)
		.def("setMat3", &Backend::Shader::setMat3)
		.def("setMat4", &Backend::Shader::setMat4);

	// Backend::Texture
	py::class_<Backend::Texture>(m, "Texture")
		.def("bind", &Backend::Texture::bind, py::arg("slot") = 0)
		.def("unbind", &Backend::Texture::unbind)
		.def("getWidth", &Backend::Texture::getWidth)
		.def("getHeight", &Backend::Texture::getHeight);

	// Backend::GraphicsDevice
	py::class_<Backend::GraphicsDevice, std::shared_ptr<Backend::GraphicsDevice>>(m, "GraphicsDevice")
		.def("createShader", py::overload_cast<const std::string&, const std::string&>(&Backend::GraphicsDevice::createShader),
			py::return_value_policy::take_ownership)
		.def("createTexture", py::overload_cast<const std::string, bool, float, float, float>(&Backend::GraphicsDevice::createTexture),
			py::arg("path"),
			py::arg("flip") = false,
			py::arg("warpParam") = GL_REPEAT,
			py::arg("minFilterParam") = GL_LINEAR_MIPMAP_LINEAR,
			py::arg("maxFilterParam") = GL_LINEAR,
			py::return_value_policy::take_ownership);

	// Mesh data structures
	py::class_<VertexAttrib>(m, "VertexAttrib")
		.def(py::init<>())
		.def_readwrite("position", &VertexAttrib::position)
		.def_readwrite("normal", &VertexAttrib::normal)
		.def_readwrite("uv", &VertexAttrib::uv);

	py::class_<All>(m, "All")
		.def(py::init<>())
		.def_readwrite("vertexAttrib", &All::vertexAttrib)
		.def_readwrite("indices", &All::indices);

	// Primitive creation functions
	m.def("createSquareData", &Prim::createSquareData, "Create square mesh data");

	// GLM types for matrix operations
	py::class_<glm::vec3>(m, "vec3")
		.def(py::init<float, float, float>())
		.def_readwrite("x", &glm::vec3::x)
		.def_readwrite("y", &glm::vec3::y)
		.def_readwrite("z", &glm::vec3::z);

	py::class_<glm::mat4>(m, "mat4")
		.def(py::init<float>());

	// GLM helper functions
	m.def("translate", [](const glm::mat4& mat, const glm::vec3& vec) {
		return glm::translate(mat, vec);
	}, "Translate a matrix by a vector");

	m.def("scale", [](const glm::mat4& mat, const glm::vec3& vec) {
		return glm::scale(mat, vec);
	}, "Scale a matrix by a vector");

	// App class with trampoline for Python overrides
	py::class_<App, PyApp>(m, "App")
        .def(py::init<>())
		.def("initialize", &App::initialize,
			py::arg("width"),
			py::arg("height"),
			py::arg("hideUi") = false,
			py::arg("backendType") = Backend::BackendType::OpenGL)
    	.def("start", &App::start)
        .def("setup", &App::setup)
		.def("preRender", &App::preRender)
		.def("render", &App::render)
		.def("postRender", &App::postRender)
		.def("getGraphicsDevice", &App::getGraphicsDevice, py::return_value_policy::reference)
		.def("addShape", [](App* self, Backend::Shader* shader, const All& data) {
			return self->addShape(shader, std::make_unique<All>(data));
		})
		.def("checkError", &App::checkError)
		.def("getCamera", &App::getCamera, py::return_value_policy::reference)
		.def("getWidth", &App::getWidth)
		.def("getHeight", &App::getHeight);

	// Bind scene system
	bind_scene(m);
}
