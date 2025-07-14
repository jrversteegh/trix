#ifndef TRIX_MATRIX_H__
#define TRIX_MATRIX_H__

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <ranges>
#include <type_traits>
#include <utility>

#include "config.h"
#include "types.h"
#include "utils.h"
#include "vector.h"

namespace trix {

template <size_t N, size_t M, typename T = Number>
struct GenericStorage;

template <size_t N, size_t M = N, typename T = Number,
          template <size_t, size_t, typename C> typename Storage =
              GenericStorage>
  requires(N > 0 && M > 0)
struct Matrix;

template <typename M>
concept MatrixConcept = requires(M const m, size_t i, size_t j) {
  typename M::value_type;
  { m.operator[](i, j) } -> std::same_as<typename M::value_type>;
  { M::rows } -> std::convertible_to<size_t const>;
  { M::columns } -> std::convertible_to<size_t const>;
};

template <typename M>
concept MutableMatrixConcept =
    MatrixConcept<M> && requires(M m, size_t i, size_t j) {
      { m.operator[](i, j) } -> std::same_as<typename M::value_type&>;
    };

template <size_t N, size_t M, typename T>
struct MatrixStorageType {
  static constexpr size_t const rows = N;
  static constexpr size_t const columns = M;
  using value_type = T;
};

template <size_t N, size_t M>
struct MatrixAssignment {
  template <typename Self, MatrixConcept O, size_t... Is>
  constexpr void row_assign(this Self&& self, O const& o, size_t i,
                            std::index_sequence<Is...>) {
    ((self[i, Is] = o[i, Is]), ...);
  }

  template <typename Self, MatrixConcept O>
    requires(O::rows == N && O::columns == M)
  constexpr auto& operator=(this Self&& self, O const& o) {
    for (size_t i = 0; i < N; ++i) {
      self.row_assign(o, i, std::make_index_sequence<M>{});
    }
    return self;
  }
};

struct RectangularBase {};

template <size_t N, size_t M>
struct RectangularType : RectangularBase {
  template <typename F>
    requires std::same_as<std::invoke_result_t<F, size_t, size_t>, void>
  constexpr void for_each_element(F fun) {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j) {
        fun(i, j);
      }
    }
  }
};

struct SymmetricBase : RectangularBase {};

template <size_t N>
struct SymmetricType : SymmetricBase {

  template <typename F>
    requires std::same_as<std::invoke_result_t<F, size_t, size_t>, void>
  constexpr void for_each_element(F fun) {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        fun(i, j);
      }
    }
  }
};

struct DiagonalBase : SymmetricBase {};

template <size_t N>
struct DiagonalType : DiagonalBase {
  template <typename F>
    requires std::same_as<std::invoke_result_t<F, size_t, size_t>, void>
  constexpr void for_each_element(F fun) {
    for (size_t i = 0; i < N; ++i) {
      fun(i, i);
    }
  }
};

template <typename STORAGE, size_t N, size_t M, size_t SIZE,
          typename T = Number>
struct MatrixArrayStorage : MatrixStorageType<N, M, T> {
  static constexpr size_t elements = SIZE;
  template <std::convertible_to<T>... Values>
  constexpr MatrixArrayStorage(Values&&... values)
      : a_{std::forward<Values>(values)...} {};
  constexpr MatrixArrayStorage(std::array<T, elements>&& array)
      : a_{std::move(array)} {}
  constexpr MatrixArrayStorage(std::array<T, elements> const& array)
      : a_{array} {}
  template <typename It, typename Ite>
    requires std::convertible_to<typename std::iterator_traits<It>::value_type,
                                 T>
  constexpr MatrixArrayStorage(It first, Ite last) {
    for (size_t i = 0; i < SIZE; ++i) {
      if (first == last)
        break;
      a_[i] = *first++;
    }
  }
  template <std::ranges::input_range R>
  constexpr MatrixArrayStorage(std::from_range_t, R const& r)
      : MatrixArrayStorage(r.begin(), r.end()) {}
  constexpr T operator[](size_t const i, size_t const j) const {
    return a_[check_and_get_offset_(i, j)];
  }
  constexpr T& operator[](size_t const i, size_t const j) {
    return a_[check_and_get_offset_(i, j)];
  }
  constexpr size_t size() const {
    return elements;
  }

  template <typename Self>
  constexpr auto data(this Self&& self) {
    return self.a_.data();
  }

private:
  std::array<T, elements> a_{};
  static constexpr size_t check_and_get_offset_(size_t const i,
                                                size_t const j) {
    size_t offset = STORAGE::get_offset(i, j);
    assert(offset < elements);
    return offset;
  }
};

template <size_t N, size_t M, typename T>
struct GenericStorage
    : MatrixArrayStorage<GenericStorage<N, M, T>, N, M, N * M, T>,
      RectangularType<N, M> {
  using MatrixArrayStorage<GenericStorage<N, M, T>, N, M, N * M,
                           T>::MatrixArrayStorage;
  static constexpr size_t get_offset(size_t const i, size_t const j) {
    return i * M + j;
  }

#ifdef HAVE_BLAS
  template <size_t K>
    requires(std::is_same_v<T, double>)
  auto blas_mul(GenericStorage<M, K, T> const& other) const {
    Matrix<N, K, double, GenericStorage> result;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, M, 1.0,
                this->data(), M, other.data(), K, 0.0, result.data(), K);
    return result;
  }
#endif
};

template <size_t N, size_t M = N, typename T = Number>
  requires(M == N)
struct SymmetricStorage
    : MatrixArrayStorage<SymmetricStorage<N, M, T>, N, M, N*(N + 1) / 2, T>,
      SymmetricType<N> {
  using MatrixArrayStorage<SymmetricStorage<N, M, T>, N, M, N*(N + 1) / 2,
                           T>::MatrixArrayStorage;
  static constexpr size_t get_offset(size_t const i, size_t const j) {
    return i > j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
  }
};

template <size_t N, size_t M = N, typename T = Number>
  requires(M == N)
struct DiagonalStorage
    : MatrixArrayStorage<DiagonalStorage<N, M, T>, N, M, N, T>,
      DiagonalType<N> {
  using Base = MatrixArrayStorage<DiagonalStorage<N, M, T>, N, M, N, T>;
  using Base::Base;
  static constexpr size_t get_offset(size_t const i,
                                     [[maybe_unused]] size_t const j) {
    return i;
  }
  constexpr T operator[](size_t const i, size_t const j) const {
    return i == j ? Base::operator[](i, j) : 0;
  }
  constexpr T& operator[](size_t const i, size_t const j) {
    return i == j ? Base::operator[](i, j) : zero_;
  }

  template <typename F>
    requires std::same_as<std::invoke_result_t<F, size_t, size_t>, void>
  constexpr void for_each_element(F fun) {
    for (size_t i = 0; i < N; ++i) {
      fun(i, i);
    }
  }

private:
  T zero_;
};

template <size_t N, size_t M = N, typename T = Number>
  requires(M == N)
struct IdentityStorage : DiagonalType<N>, MatrixStorageType<N, N, T> {
  static constexpr size_t const rows = N;
  static constexpr size_t const columns = N;
  constexpr T operator[](size_t const i, size_t const j) const {
    return i == j ? 1 : 0;
  }
};

template <MatrixConcept X>
struct MatrixView {
  explicit constexpr MatrixView(X& x) : matrix{x} {};
  X& matrix;
};

template <MatrixConcept X>
struct Transpose
    : MatrixView<X>,
      MatrixStorageType<X::columns, X::rows, typename X::value_type> {
  using MatrixView<X>::MatrixView;

  constexpr auto& operator[](size_t const i, size_t const j) {
    return this->matrix[j, i];
  }
  constexpr auto operator[](size_t const i, size_t const j) const {
    return this->matrix[j, i];
  }
};

template <MatrixConcept X, size_t I, size_t J, size_t N, size_t M>
struct SubMatrix : MatrixView<X>,
                   MatrixStorageType<N, M, typename X::value_type>,
                   MatrixAssignment<N, M> {
  using MatrixView<X>::MatrixView;
  using MatrixAssignment<N, M>::operator=;
  constexpr auto& operator[](size_t const i, size_t const j) {
    static typename X::value_type zero = 0;
    assert(i < N);
    assert(j < M);
    size_t p = I + i;
    size_t q = J + j;
    if (p < X::rows && q < X::columns) {
      return this->matrix[p, q];
    } else {
      return zero;
    }
  }
  constexpr auto operator[](size_t const i, size_t const j) const {
    assert(i < N);
    assert(j < M);
    size_t p = I + i;
    size_t q = J + j;
    if (p < X::rows && q < X::columns) {
      return this->matrix[p, q];
    } else {
      return static_cast<typename X::value_type>(0);
    }
  }
};

template <MatrixConcept X>
struct Diagonal
    : MatrixView<X>,
      VectorStorageType<std::min(X::rows, X::columns), typename X::value_type> {
  using MatrixView<X>::MatrixView;

  constexpr auto& operator[](size_t const i) {
    return this->matrix[i, i];
  }
  constexpr auto operator[](size_t const i) const {
    return this->matrix[i, i];
  }
};

template <MatrixConcept X, size_t S>
struct RowColumnView : MatrixView<X> {
  constexpr RowColumnView(X& x, size_t const i) : MatrixView<X>{x}, index{i} {
    assert(i < S);
  };
  size_t const index;
};

template <MatrixConcept X, size_t S>
struct Row : RowColumnView<X, S>,
             VectorStorageType<X::columns, typename X::value_type> {
  using RowColumnView<X, S>::RowColumnView;
  constexpr auto& operator[](size_t const i) {
    return this->matrix[this->index, i];
  }
  constexpr auto operator[](size_t const i) const {
    return this->matrix[this->index, i];
  }
};

template <MatrixConcept X, size_t S>
struct Column : RowColumnView<X, S>,
                VectorStorageType<X::rows, typename X::value_type> {
  using RowColumnView<X, S>::RowColumnView;
  constexpr auto& operator[](size_t const i) {
    return this->matrix[i, this->index];
  }
  constexpr auto operator[](size_t const i) const {
    return this->matrix[i, this->index];
  }
};

template <size_t N, size_t M, typename T,
          template <size_t, size_t, typename C> typename Storage>
  requires(N > 0 && M > 0)
struct Matrix : Storage<N, M, T> {
  using Storage<N, M, T>::Storage;
  template <MatrixConcept O>
  explicit constexpr Matrix(O const& other)
      : Storage<O::rows, O::columns, typename O::value_type>{} {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] = other[i, j]; });
  }

  template <MatrixConcept O>
    requires(O::rows == N && O::columns == M)
  constexpr Matrix& operator+=(O const& other) {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] += other[i, j]; });
    return *this;
  }

  template <MatrixConcept O>
    requires(O::rows == N && O::columns == M)
  constexpr Matrix& operator-=(O const& other) {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] -= other[i, j]; });
    return *this;
  }

  constexpr Matrix& operator*=(ScalarConcept auto const value) {
    this->for_each_element(
        [this, value](size_t i, size_t j) { (*this)[i, j] *= value; });
    return *this;
  }

  template <typename Self>
  constexpr auto& transpose(this Self&& self) {
    static auto result = Transpose<std::remove_reference_t<Self>>{self};
    return result;
  }

  template <typename Self>
  constexpr auto& diagonal(this Self&& self) {
    static auto result = Diagonal<std::remove_reference_t<Self>>{self};
    return result;
  }

  template <typename Self>
  constexpr auto& row(this Self&& self, size_t const i) {
    static auto result = Row<std::remove_reference_t<Self>, self.rows>{self, i};
    return result;
  }

  template <typename Self>
  constexpr auto& column(this Self&& self, size_t const i) {
    static auto result =
        Column<std::remove_reference_t<Self>, self.columns>{self, i};
    return result;
  }
};

template <size_t N, typename T = Number>
using SymmetricMatrix = Matrix<N, N, T, SymmetricStorage>;

template <size_t N, typename T = Number>
using DiagonalMatrix = Matrix<N, N, T, DiagonalStorage>;

template <size_t N, typename T = Number>
using IdentityMatrix = Matrix<N, N, T, IdentityStorage>;

template <typename C, typename... Cs>
auto matrix(C&& first, Cs&&... components) {
  constexpr size_t S = static_cast<size_t>(size_t_sqrt<sizeof...(Cs) + 1>());
  return Matrix<S, S, C>{std::forward<C>(first),
                         std::forward<Cs>(components)...};
}

template <MatrixConcept X1, MatrixConcept X2, size_t... Is>
constexpr bool row_equals(X1 const& x1, X2 const& x2, size_t i,
                          std::index_sequence<Is...>) {
  return (... && (x1[i, Is] == x2[i, Is]));
}

template <MatrixConcept X1, MatrixConcept X2, size_t... Is>
constexpr bool equals(X1 const& x1, X2 const& x2, std::index_sequence<Is...>) {
  return (... &&
          (row_equals(x1, x2, Is, std::make_index_sequence<X1::columns>{})));
}

template <MatrixConcept X1, MatrixConcept X2>
  requires(X1::rows == X2::rows && X1::columns == X2::columns)
constexpr bool operator==(X1 const& x1, X2 const& x2) {
  return equals(x1, x2, std::make_index_sequence<X1::rows>{});
}

template <MatrixConcept X1, MatrixConcept X2>
  requires(X1::rows == X2::rows && X1::columns == X2::columns)
constexpr auto operator+(X1 const& x1, X2 const& x2) {
  Matrix<X1::rows, X1::columns,
         std::common_type_t<typename X1::value_type, typename X2::value_type>>
      result{x1};
  result += x2;
  return result;
}

template <size_t N, typename T1, typename T2,
          template <size_t, size_t, typename> typename S1,
          template <size_t, size_t, typename> typename S2>
  requires std::derived_from<Matrix<N, N, T1, S1>, SymmetricBase> &&
           std::derived_from<Matrix<N, N, T2, S2>, SymmetricBase>
constexpr auto operator+(Matrix<N, N, T1, S1> const& x1,
                         Matrix<N, N, T2, S2> const& x2) {
  SymmetricMatrix<N, std::common_type_t<T1, T2>> result{x1};
  result += x2;
  return result;
}

template <size_t N, typename T1, typename T2>
constexpr auto operator+(DiagonalMatrix<N, T1> const& x1,
                         DiagonalMatrix<N, T2> const& x2) {
  DiagonalMatrix<N, std::common_type_t<T1, T2>> result{x1};
  result += x2;
  return result;
}

template <MatrixConcept X1, MatrixConcept X2>
  requires(X1::rows == X2::rows && X1::columns == X2::columns)
constexpr auto operator-(X1 const& x1, X2 const& x2) {
  Matrix<X1::rows, X1::columns,
         std::common_type_t<typename X1::value_type, typename X2::value_type>>
      result{x1};
  result -= x2;
  return result;
}

template <size_t N, typename T1, typename T2,
          template <size_t, size_t, typename> typename S1,
          template <size_t, size_t, typename> typename S2>
  requires std::derived_from<Matrix<N, N, T1, S1>, SymmetricBase> &&
           std::derived_from<Matrix<N, N, T2, S2>, SymmetricBase>
constexpr auto operator-(Matrix<N, N, T1, S1> const& x1,
                         Matrix<N, N, T2, S2> const& x2) {
  SymmetricMatrix<N, std::common_type_t<T1, T2>> result{x1};
  result -= x2;
  return result;
}

template <size_t N, typename T1, typename T2>
constexpr auto operator-(DiagonalMatrix<N, T1> const& x1,
                         DiagonalMatrix<N, T2> const& x2) {
  DiagonalMatrix<N, std::common_type_t<T1, T2>> result{x1};
  result -= x2;
  return result;
}

template <size_t N, size_t M, typename T,
          template <size_t, size_t, typename> typename S, ScalarConcept V>
  requires MutableMatrixConcept<Matrix<N, M, T, S>>
constexpr auto operator*(Matrix<N, M, T, S> const& x, V const value) {
  Matrix<N, M, T, S> result{x};
  result *= value;
  return result;
}

template <MatrixConcept X, ScalarConcept V>
  requires std::derived_from<DiagonalBase, X>
constexpr auto operator*(X const& x, V const value) {
  DiagonalMatrix<X::rows, typename X::value_type> result{x};
  result *= value;
  return result;
}

template <MatrixConcept X, ScalarConcept V>
constexpr auto operator*(X const& x, V const value) {
  Matrix<X::rows, X::columns, typename X::value_type> result{x};
  result *= value;
  return result;
}

template <MatrixConcept X, ScalarConcept V>
constexpr auto operator*(V const value, X const& x) {
  return x * value;
}

template <MatrixConcept X1, MatrixConcept X2, size_t... Is>
constexpr auto row_column_mul(X1 const& x1, X2 const& x2, size_t i, size_t j,
                              std::index_sequence<Is...>) {
  if constexpr (sizeof...(Is) < 32) {
    return (... + (x1[i, Is] * x2[Is, j]));
  } else {
    auto result = x1[i, 0] * x2[0, j];
    for (size_t k = 1; k < sizeof...(Is); ++k) {
      result += x1[i, k] * x2[k, j];
    }
    return result;
  }
}

template <MatrixConcept X1, MatrixConcept X2>
  requires(X1::columns == X2::rows)
constexpr auto operator*(X1 const& x1, X2 const& x2) {
  Matrix<X1::rows, X2::columns,
         std::common_type_t<typename X1::value_type, typename X2::value_type>>
      result;
  for (size_t i = 0; i < X1::rows; ++i) {
    for (size_t j = 0; j < X2::columns; ++j) {
      result[i, j] = x1[i, 0] * x2[0, j];
      for (size_t k = 1; k < X1::columns; ++k) {
        result[i, j] += x1[i, k] * x2[k, j];
      }
    }
  }
  return result;
}

#ifndef __clang__
template <size_t N, size_t K, size_t M, typename T1, typename T2>
constexpr auto operator*(Matrix<N, K, T1> const& x1,
                         Matrix<K, M, T2> const& x2) {
  Matrix<N, M, std::common_type_t<T1, T2>> result;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      result[i, j] =
          row_column_mul(x1, x2, i, j, std::make_index_sequence<K>{});
    }
  }
  return result;
}
#endif

#ifdef HAVE_BLAS
template <size_t N, size_t K, size_t M>
constexpr auto blas_mul(GenericStorage<N, K, double> const& x1,
                        GenericStorage<K, M, double> const& x2) {
  auto result = x1.blas_mul(x2);
  return result;
}
#endif

template <size_t N, typename T1, typename T2>
constexpr auto operator*(SymmetricMatrix<N, T1> const& x1,
                         SymmetricMatrix<N, T2> const& x2) {
  SymmetricMatrix<N, std::common_type_t<T1, T2>> result;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      result[i, j] =
          row_column_mul(x1, x2, i, j, std::make_index_sequence<N>{});
    }
  }
  return result;
}

template <size_t N, typename T1, typename T2>
constexpr auto operator*(DiagonalMatrix<N, T1> const& x1,
                         DiagonalMatrix<N, T2> const& x2) {
  DiagonalMatrix<N, std::common_type_t<T1, T2>> result;
  for (size_t i = 0; i < N; ++i) {
    result[i, i] = x1[i, i] * x2[i, i];
  }
  return result;
}

template <MatrixConcept X, VectorConcept V,
          typename T = std::common_type_t<typename X::value_type,
                                          typename V::value_type>>
  requires(X::columns == V::components)
constexpr auto operator*(X const& x, V const& v) {
  Vector<X::rows, T> result;
  for (size_t i = 0; i < X::rows; ++i) {
    result[i] = x[i, 0] * v[0];
    for (size_t j = 1; j < V::components; ++j) {
      result[i] += x[i, j] * v[j];
    }
  }
  return result;
}

template <MatrixConcept X, VectorConcept V,
          typename T = std::common_type_t<typename X::value_type,
                                          typename V::value_type>>
  requires(X::rows == V::components)
constexpr auto operator*(V const& v, X const& x) {
  Vector<X::columns, T> result;
  for (size_t i = 0; i < X::columns; ++i) {
    result[i] = v[0] * x[0, i];
    for (size_t j = 1; j < V::components; ++j) {
      result[i] += v[j] * x[j, i];
    }
  }
  return result;
}

template <MatrixConcept X1, MatrixConcept X2>
  requires(X1::rows == X2::rows && X1::columns == X2::columns)
constexpr bool
all_close(X1 const& x1, X2 const& x2,
          typename X1::value_type threshold =
              1000 * std::numeric_limits<typename X1::value_type>::epsilon()) {
  for (size_t i = 0; i < X1::rows; ++i) {
    for (size_t j = 0; j < X1::columns; ++j) {
      if (std::abs(x1[i, j] - x2[i, j]) > threshold)
        return false;
    }
  }
  return true;
}

template <size_t MinSize = 16>
  requires(MinSize > 0)
struct Strassen {
  template <MatrixConcept X1, MatrixConcept X2>
    requires(X1::columns == X2::rows)
  static constexpr auto operator()(X1 const& x1, X2 const& x2) {
    if constexpr (X1::rows <= MinSize || X1::columns <= MinSize ||
                  X2::columns <= MinSize) {
      return x1 * x2;
    } else {
      static constexpr size_t N =
          std::max((X1::rows + 1) / 2, (X1::columns + 1) / 2);
      using result_type = Matrix<
          X1::rows, X2::columns,
          std::common_type_t<typename X1::value_type, typename X2::value_type>>;
      result_type result;
      auto a11 = SubMatrix<X1 const, 0, 0, N, N>{x1};
      auto a12 = SubMatrix<X1 const, 0, N, N, N>{x1};
      auto a21 = SubMatrix<X1 const, N, 0, N, N>{x1};
      auto a22 = SubMatrix<X1 const, N, N, N, N>{x1};
      auto b11 = SubMatrix<X2 const, 0, 0, N, N>{x2};
      auto b12 = SubMatrix<X2 const, 0, N, N, N>{x2};
      auto b21 = SubMatrix<X2 const, N, 0, N, N>{x2};
      auto b22 = SubMatrix<X2 const, N, N, N, N>{x2};
      auto r11 = SubMatrix<result_type, 0, 0, N, N>{result};
      auto r12 = SubMatrix<result_type, 0, N, N, N>{result};
      auto r21 = SubMatrix<result_type, N, 0, N, N>{result};
      auto r22 = SubMatrix<result_type, N, N, N, N>{result};
      auto m1 = Strassen::operator()((a11 + a22), (b11 + b22));
      auto m2 = Strassen::operator()(a21 + a22, b11);
      auto m3 = Strassen::operator()(a11, b12 - b22);
      auto m4 = Strassen::operator()(a22, b21 - b11);
      auto m5 = Strassen::operator()(a11 + a12, b22);
      r11 = m1 + m4 - m5 + Strassen::operator()(a12 - a22, b21 + b22);
      r12 = m3 + m5;
      r21 = m2 + m4;
      r22 = m1 - m2 + m3 + Strassen::operator()(a21 - a11, b11 + b12);
      return result;
    }
  }
};

template <size_t MinSize = 16>
  requires(MinSize > 0)
struct BlockMul {
  template <MatrixConcept X1, MatrixConcept X2>
    requires(X1::columns == X2::rows)
  static constexpr auto operator()(X1 const& x1, X2 const& x2) {
    if constexpr (X1::rows <= MinSize || X1::columns <= MinSize ||
                  X2::columns <= MinSize) {
      return x1 * x2;
    } else {
      static constexpr size_t N =
          std::max((X1::rows + 1) / 2, (X1::columns + 1) / 2);
      using result_type = Matrix<
          X1::rows, X2::columns,
          std::common_type_t<typename X1::value_type, typename X2::value_type>>;
      result_type result;
      auto a11 = SubMatrix<X1 const, 0, 0, N, N>{x1};
      auto a12 = SubMatrix<X1 const, 0, N, N, N>{x1};
      auto a21 = SubMatrix<X1 const, N, 0, N, N>{x1};
      auto a22 = SubMatrix<X1 const, N, N, N, N>{x1};
      auto b11 = SubMatrix<X2 const, 0, 0, N, N>{x2};
      auto b12 = SubMatrix<X2 const, 0, N, N, N>{x2};
      auto b21 = SubMatrix<X2 const, N, 0, N, N>{x2};
      auto b22 = SubMatrix<X2 const, N, N, N, N>{x2};
      auto r11 = SubMatrix<result_type, 0, 0, N, N>{result};
      auto r12 = SubMatrix<result_type, 0, N, N, N>{result};
      auto r21 = SubMatrix<result_type, N, 0, N, N>{result};
      auto r22 = SubMatrix<result_type, N, N, N, N>{result};
      r11 = BlockMul::operator()(a11, b11) + BlockMul::operator()(a12, b21);
      r12 = BlockMul::operator()(a11, b12) + BlockMul::operator()(a12, b22);
      r21 = BlockMul::operator()(a21, b11) + BlockMul::operator()(a22, b21);
      r22 = BlockMul::operator()(a21, b12) + BlockMul::operator()(a22, b22);
      return result;
    }
  }
};

static constexpr auto strassen = Strassen<>{};
static constexpr auto block_mul = BlockMul<>{};

} // namespace trix

#endif
