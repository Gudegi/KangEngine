/*#include "kangEngine.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(kangengine, m) {
    py::class_<App>(m, "App")
        .def(py::init<>());

    m.def("setup",		&App::setup);
	m.def("preRender",	&App::preRender);
	m.def("render",		&App::render);
	m.def("postRender",	&App::postRender);
    //m.def("call_go", &call_go);
}*/

#include <pybind11/pybind11.h>
namespace py = pybind11;
int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(kangengine, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}