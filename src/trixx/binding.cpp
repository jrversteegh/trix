/**
 * \file
 * \brief Provide python bindings for C++ objects and functions
 *
 * \author J.R. Versteegh <j.r.versteegh@gmail.com>
 */

#include <pybind11/pybind11.h>

#include "trix/matrix.h"
#include "trix/printing.h"

namespace py = pybind11;
using namespace trix;

using Matrix33 = Matrix<3, 3>;


PYBIND11_MODULE(trixx, m) {
  m.doc() = "C++ trix module";
  py::class_<Matrix33>(m, "Matrix33")
      .def(py::init<Number, Number, Number, Number, Number, Number, Number, Number, Number>())
      .def("__str__", [](Matrix33 const& self ){ return to_string(self); });
}

