#include <pybind11/pybind11.h>
#include "KinfuPlarr.h"
#include <pybind11/stl.h>
//#include "ndarray_converter.h"

namespace py = pybind11;

PYBIND11_MODULE(kinfu_cv, m) {

    // NDArrayConverter::init_numpy();

    // optional module docstring
    m.doc() = "pybind11 kinfu_cv plugin";

    // define add function
    //m.def("add", &add, "A function which adds two numbers");

    // bindings to Pet class
    py::class_<KinfuPlarr>(m, "KinfuPlarr")
            .def(py::init<unsigned int, unsigned int, float, float, float, float, float, bool>())
            .def("integrateFrame", &KinfuPlarr::integrateFrame)
            .def("renderShow", &KinfuPlarr::renderShow)
            .def("reset", &KinfuPlarr::reset)
            .def("getPose", &KinfuPlarr::getPose)
            .def("getCurrentDepth", &KinfuPlarr::getCurrentDepth);
}