#include "matrix.hpp"
#include "permanent.hpp"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <complex>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <type_traits>
#include <utility>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
namespace ffi = xla::ffi;

template <ffi::DataType T>
std::pair<int64_t, int64_t> get_dims(const ffi::Buffer<T> &buffer)
{
  auto dims = buffer.dimensions();

  if (dims.size() == 0)
  {
    return std::make_pair(0, 0);
  }
  return std::make_pair(buffer.element_count(), dims.back());
}

ffi::Error PermanentImpl(ffi::Buffer<ffi::C128> A, ffi::Buffer<ffi::U64> rows,
                         ffi::Buffer<ffi::U64> cols,
                         ffi::ResultBuffer<ffi::C128> y)
{
  auto [total_size, n] = get_dims(A);

  if (n == 0)
  {
    return ffi::Error::InvalidArgument("Perm input must be a matrix");
  }
  std::vector<int> row_mult(&(rows.typed_data()[0]),
                            &(rows.typed_data()[0]) + total_size / n);
  std::vector<int> col_mult(&(cols.typed_data()[0]),
                            &(cols.typed_data()[0]) + n);

  Matrix<std::complex<double>> matrix(total_size / n, n, &(A.typed_data()[0]));

  y->typed_data()[0] =
      permanent<std::complex<double>, double>(matrix, row_mult, col_mult);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(Permanent, PermanentImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Ret<ffi::Buffer<ffi::C128>>());

ffi::Error PermFwdImpl(ffi::Buffer<ffi::C128> A, ffi::Buffer<ffi::U64> rows,
                       ffi::Buffer<ffi::U64> cols,
                       ffi::ResultBuffer<ffi::C128> y,
                       ffi::ResultBuffer<ffi::C128> res)
{
  auto [total_size, n] = get_dims(A);
  if (n == 0)
  {
    return ffi::Error::InvalidArgument("Permanent input must be a matrix");
  }

  std::vector<int> row_mult(rows.typed_data(), rows.typed_data() + n);
  std::vector<int> col_mult(cols.typed_data(), cols.typed_data() + n);

  Matrix<std::complex<double>> matrix(total_size / n, n, &(A.typed_data()[0]));

  y->typed_data()[0] =
      permanent<std::complex<double>, double>(matrix, row_mult, col_mult);

  res->typed_data()[0] =
      permanent<std::complex<double>, double>(matrix, row_mult, col_mult);
  ;

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PermFwd, PermFwdImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Ret<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>());

void ComputePermBwd(std::complex<double> res, Matrix<std::complex<double>> &A,
                    std::vector<int> &rows, std::vector<int> &cols,
                    const std::vector<std::complex<double>> &cotangent,
                    std::complex<double> *ct_x)
{

  Matrix<std::complex<double>> grad = grad_perm(A, rows, cols);

  for (int64_t i = 0; i < grad.rows; ++i)
  {
    for (int64_t j = 0; j < grad.cols; ++j)
    {
      //ct_x[i * A.cols + j] = cotangent * grad(i, j);

      std::cout << cotangent.at(i) << " * " << grad(i, j) << std::endl;
      ct_x[i * A.cols + j] = cotangent.at(i) * grad(i, j);
    }
  }
}

ffi::Error PermBwdImpl(ffi::Buffer<ffi::C128> res, ffi::Buffer<ffi::C128> A,
                       ffi::Buffer<ffi::U64> rows, ffi::Buffer<ffi::U64> cols,
                       ffi::Buffer<ffi::C128> cotangent,
                       ffi::ResultBuffer<ffi::C128> ct_x)
{
  auto [total_size, n] = get_dims(A);
  if (n == 0)
  {
    return ffi::Error::InvalidArgument("RmsNormBwd inputs must be arrays");
  }

  std::vector<int> row_mult(rows.typed_data(), rows.typed_data() + n);
  std::vector<int> col_mult(cols.typed_data(), cols.typed_data() + n);

  Matrix<std::complex<double>> matrix(total_size / n, n, &(A.typed_data()[0]));

  std::vector<std::complex<double>> cot_vector(cotangent.typed_data(), cotangent.typed_data() + n);

  ComputePermBwd(res.typed_data()[0], matrix, row_mult, col_mult,
                  cot_vector,
                 &(ct_x->typed_data()[0]));

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PermBwd, PermBwdImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::C128>>() // res
                                  .Arg<ffi::Buffer<ffi::C128>>() // A
                                  .Arg<ffi::Buffer<ffi::U64>>()  // rows
                                  .Arg<ffi::Buffer<ffi::U64>>()  // cols
                                  .Arg<ffi::Buffer<ffi::C128>>() // cotangent
                                  .Ret<ffi::Buffer<ffi::C128>>() // ct_x
);

template <typename T>
py::capsule EncapsulateFfiHandler(T *fn)
{
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}

PYBIND11_MODULE(_core, m)
{
  m.doc() = R"pbdoc(
        Permanent calculator plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           permanent
    )pbdoc";
  m.def("registrations", []()
        {
    py::dict registrations;
    registrations["perm"] = EncapsulateFfiHandler(Permanent);
    registrations["perm_fwd"] = EncapsulateFfiHandler(PermFwd);
    registrations["perm_bwd"] = EncapsulateFfiHandler(PermBwd);
    return registrations; });

  m.attr("__version__") = "dev";
}
