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
//https://github.com/caiorss/example-pybind11-vcpkg/blob/master/cashflow.cpp
class CashFlow{
private:
	std::vector<double> m_flow;
public:
	CashFlow(){}
	
	void add(double x){
		m_flow.push_back(x);      
	}
	void addVector(std::vector<double> const& xs){
		for(const auto& x: xs)
			m_flow.push_back(x);
	}	
	void clear(){
		m_flow.clear();
	}
	
	int size(){
		return m_flow.size();
	}
	double get(int index){
		return m_flow[index];
	}
	double npv(double rate){
		double rr   = 1 + rate;
		double prod = 1.0;
		double acc  = 0;
		for(auto x: m_flow){
			acc  += x / prod;
			prod *= rr;
		}
		return acc;
	}
	void show(){
		int i = 0;
		for(auto x: m_flow){
			std::cout << "i  = " << i
					  << " c  = " << std::setw(10)
					  << std::fixed << std::setprecision(2) << x
					  << std::endl;
			i++;
		}
	}
};

PYBIND11_MODULE(kangengine, m) {
	// optional module docstring
    m.doc() = "pybind11 sample CashFlow module."; 
	// Add function to module.
    m.def("add", &add, "A function which adds two numbers");

	auto pyCashFlow = py::class_<CashFlow>(
			m,
			"CashFlow",
			R"pbdoc(Class for performing cash flow computations.)pbdoc"
			);
	pyCashFlow
		.def(py::init<>())             // Bind class constructor 
		.def("add",			&CashFlow::add)
		.def("addVector",	&CashFlow::addVector)
		.def("size",		&CashFlow::size)
		.def("get",			&CashFlow::get)
     	.def("show",		&CashFlow::show)
		.def("clear",		&CashFlow::clear)
		.def("npv",			&CashFlow::npv);

    
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
}
