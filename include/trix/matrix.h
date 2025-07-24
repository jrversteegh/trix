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

template <typename X>
concept MatrixConcept = requires(noref<X> const x, size_t i, size_t j) {
  typename noref<X>::value_type;
  typename noref<X>::result_type;
  { x.operator[](i, j) } -> std::same_as<typename noref<X>::value_type>;
  { noref<X>::rows } -> std::convertible_to<size_t const>;
  { noref<X>::columns } -> std::convertible_to<size_t const>;
};

template <typename X>
concept MutableMatrixConcept =
    MatrixConcept<X> && requires(X x, size_t i, size_t j) {
      { x.operator[](i, j) } -> std::same_as<typename X::value_type&>;
    };

template <size_t N, size_t M, typename T>
struct MatrixStorageType {
  static constexpr size_t const rows = N;
  static constexpr size_t const columns = M;
  using value_type = T;
};

template <size_t N, size_t M>
struct MatrixAssignment {
  template <typename Self, MatrixConcept O, size_t... Js>
  constexpr void row_assign(this Self& self, O const& o, size_t i,
                            std::index_sequence<Js...>) {
    ((self[i, Js] = o[i, Js]), ...);
  }

  template <typename Self, MatrixConcept O>
    requires(O::rows == N && O::columns == M)
  constexpr auto& operator=(this Self& self, O const& o) {
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
    requires std::same_as<std::invoke_result_t<F, size_t const, size_t const>,
                          void>
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
    requires std::same_as<std::invoke_result_t<F, size_t const, size_t const>,
                          void>
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
    requires std::same_as<std::invoke_result_t<F, size_t const, size_t const>,
                          void>
  constexpr void for_each_element(F fun) {
    for (size_t i = 0; i < N; ++i) {
      fun(i, i);
    }
  }
};

template <typename Impl, size_t N, size_t M, size_t S, typename T = Number>
struct MatrixArrayStorage : MatrixStorageType<N, M, T> {
  static constexpr size_t elements = S;
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
    for (size_t i = 0; i < elements; ++i) {
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
    size_t offset = Impl::get_offset(i, j);
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
    : MatrixArrayStorage<SymmetricStorage<N, N, T>, N, N, N*(N + 1) / 2, T>,
      SymmetricType<N> {
  using MatrixArrayStorage<SymmetricStorage<N, N, T>, N, N, N*(N + 1) / 2,
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

template <size_t I>
struct RowAt {
  template <MatrixConcept X1, MatrixConcept X2, size_t... Js>
  static constexpr bool equals(X1 const& x1, X2 const& x2,
                               std::index_sequence<Js...>) {
    return (... && (x1[I, Js] == x2[I, Js]));
  }
  template <MatrixConcept X1, MatrixConcept X2, typename value_type,
            size_t... Js>
  static constexpr bool close(X1 const& x1, X2 const& x2, value_type threshold,
                              std::index_sequence<Js...>) {
    return (... && (std::abs(x1[I, Js] - x2[I, Js]) < threshold));
  }
};

template <MatrixConcept X1, MatrixConcept X2>
struct CommonMatrix {
  using value_type =
      std::common_type_t<typename X1::value_type, typename X2::value_type>;
  using type = std::conditional_t<
      std::is_convertible_v<typename X1::result_type, DiagonalBase> &&
          std::is_convertible_v<typename X2::result_type, DiagonalBase>,
      Matrix<X1::rows, X2::columns, value_type, DiagonalStorage>,
      std::conditional_t<
          std::is_convertible_v<typename X1::result_type, SymmetricBase> &&
              std::is_convertible_v<typename X2::result_type, SymmetricBase>,
          Matrix<X1::rows, X2::columns, value_type, SymmetricStorage>,
          Matrix<X1::rows, X2::columns, value_type>>>;
};

template <MatrixConcept X1, MatrixConcept X2>
using common_matrix_t = CommonMatrix<typename noref<X1>::result_type,
                                     typename noref<X2>::result_type>::type;

struct ConstructorTag {};

template <typename R>
struct ResultType {
  using result_type = R;
  template <class Self>
  explicit constexpr operator result_type(this Self&& self) {
    return result_type{std::forward<Self>(self), ConstructorTag{}};
  }
  template <class Self>
  constexpr auto operator()(this Self&& self) {
    return static_cast<result_type>(std::forward<Self>(self));
  }
};

template <MatrixConcept X, typename R = typename X::result_type>
struct MatrixUnaryOperation
    : MatrixStorageType<R::rows, R::columns, typename R::value_type>,
      ResultType<R> {
  using arg_type = make_const<X>;
  MatrixUnaryOperation() = delete;
  template <typename U>
  explicit constexpr MatrixUnaryOperation(U&& arg)
      : arg{std::forward<U>(arg)} {}
  arg_type arg;
};

template <typename T>
struct UseResult {
  using type = T;
};

template <typename T>
  requires MatrixConcept<T> || VectorConcept<T>
struct UseResult<T> {
  using type = std::conditional_t<
      std::is_same_v<noref<T>, typename noref<T>::result_type>, T,
      typename noref<T>::result_type>;
};

template <typename T>
using use_result_t = UseResult<T>::type;

template <MatrixConcept X1, typename X2, typename R,
          bool evaluate_intermediates = false>
struct MatrixBinaryOperation : ResultType<R> {
  using arg1_type = make_const<
      std::conditional_t<evaluate_intermediates, use_result_t<X1>, X1>>;
  using arg2_type = make_const<
      std::conditional_t<evaluate_intermediates, use_result_t<X2>, X2>>;
  MatrixBinaryOperation() = delete;
  template <typename U1, typename U2>
  constexpr MatrixBinaryOperation(U1&& arg1, U2&& arg2)
      : arg1{std::forward<U1>(arg1)}, arg2{std::forward<U2>(arg2)} {}
  arg1_type arg1;
  arg2_type arg2;
};

template <MatrixConcept X1, MatrixConcept X2,
          bool evaluate_intermediates = false,
          typename R = common_matrix_t<X1, X2>>
struct MatrixMatrixOperation
    : MatrixBinaryOperation<X1, X2, R, evaluate_intermediates>,
      MatrixStorageType<R::rows, R::columns, typename R::value_type> {
  using MatrixBinaryOperation<X1, X2, R,
                              evaluate_intermediates>::MatrixBinaryOperation;
};

template <MatrixConcept X, VectorConcept V, bool reversed,
          typename R = std::conditional_t<
              reversed,
              Vector<noref<X>::columns,
                     std::common_type_t<typename noref<X>::value_type,
                                        typename noref<V>::value_type>>,
              Vector<noref<X>::rows,
                     std::common_type_t<typename noref<X>::value_type,
                                        typename noref<V>::value_type>>>>
struct MatrixVectorOperation
    : MatrixBinaryOperation<X, V, R>,
      VectorStorageType<R::components, typename R::value_type> {
  using MatrixBinaryOperation<X, V, R>::MatrixBinaryOperation;
};

template <MatrixConcept X, ScalarConcept S,
          typename R = typename noref<X>::result_type::template retyped<
              std::common_type_t<typename noref<X>::value_type, S>>>
struct MatrixScalarOperation
    : MatrixBinaryOperation<X, S, R>,
      MatrixStorageType<R::rows, R::columns, typename R::value_type> {
  using MatrixBinaryOperation<X, S, R>::MatrixBinaryOperation;
};

template <MatrixConcept X>
struct MatrixNegation : MatrixUnaryOperation<X> {
  using MatrixUnaryOperation<X>::MatrixUnaryOperation;
  constexpr auto operator[](size_t const i, size_t const j) const {
    return -(this->arg[i, j]);
  }
};

template <MatrixConcept X1, MatrixConcept X2>
struct MatrixAddition : MatrixMatrixOperation<X1, X2> {
  using MatrixMatrixOperation<X1, X2>::MatrixMatrixOperation;
  constexpr auto operator[](size_t const i, size_t const j) const {
    return this->arg1[i, j] + this->arg2[i, j];
  }
};

template <MatrixConcept X1, MatrixConcept X2>
struct MatrixSubtraction : MatrixMatrixOperation<X1, X2> {
  using MatrixMatrixOperation<X1, X2>::MatrixMatrixOperation;
  constexpr auto operator[](size_t const i, size_t const j) const {
    return this->arg1[i, j] - this->arg2[i, j];
  }
};

template <MatrixConcept X1, MatrixConcept X2>
struct MatrixMultiplication : MatrixMatrixOperation<X1, X2, true> {
  using MatrixMatrixOperation<X1, X2, true>::MatrixMatrixOperation;
  template <size_t I, size_t J, size_t... Ks>
  constexpr auto dot(std::index_sequence<Ks...>) const {
    return ((this->arg1[I, Ks] * this->arg2[Ks, J]) + ...);
  }

  template <size_t I, size_t J>
  constexpr auto get() const {
    return dot<I, J>(std::make_index_sequence<noref<X1>::columns>{});
  }

  constexpr auto operator[](size_t const i, size_t const j) const {
    auto result = this->arg1[i, 0] * this->arg2[0, j];
    for (size_t k = 1; k < noref<X1>::columns; ++k) {
      result += this->arg1[i, k] * this->arg2[k, j];
    }
    return result;
  }
};

template <MatrixConcept X, VectorConcept V, bool reversed>
struct MatrixVectorMultiplication : MatrixVectorOperation<X, V, reversed> {
  using MatrixVectorOperation<X, V, reversed>::MatrixVectorOperation;

  template <size_t... Is>
  constexpr auto dot(size_t const i, std::index_sequence<Is...>) const {
    if constexpr (reversed) {
      return ((this->arg1[Is, i] * this->arg2[Is]) + ...);
    } else {
      return ((this->arg1[i, Is] * this->arg2[Is]) + ...);
    }
  }

  constexpr auto operator[](size_t const i) const {
    return dot(i, std::make_index_sequence<noref<V>::components>{});
  }
};

template <MatrixConcept X, ScalarConcept S>
struct MatrixScalarMultiplication : MatrixScalarOperation<X, S> {
  using MatrixScalarOperation<X, S>::MatrixScalarOperation;

  constexpr auto operator[](size_t const i, size_t const j) const {
    return this->arg1[i, j] * this->arg2;
  }
};

template <MatrixConcept X>
struct MatrixView {
  template <typename U>
  explicit constexpr MatrixView(U&& x) : matrix{std::forward<U>(x)} {};
  X matrix;
};

template <MatrixConcept X>
struct Transpose
    : MatrixView<X>,
      MatrixStorageType<X::columns, X::rows, typename X::value_type> {
  using result_type =
      typename X::result_type::template resized<X::columns, X::rows>;
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
                   MatrixStorageType<N, M, typename noref<X>::value_type>,
                   MatrixAssignment<N, M> {
  using MatrixView<X>::MatrixView;
  using MatrixAssignment<N, M>::operator=;
  using result_type = typename noref<X>::result_type::template resized<N, M>;
  constexpr auto& operator[](size_t const i, size_t const j) {
    static typename noref<X>::value_type zero = 0;
    assert(i < N);
    assert(j < M);
    size_t p = I + i;
    size_t q = J + j;
    if (p < noref<X>::rows && q < noref<X>::columns) {
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
    if (p < noref<X>::rows && q < noref<X>::columns) {
      return std::as_const(this->matrix)[p, q];
    } else {
      return static_cast<typename noref<X>::value_type>(0);
    }
  }
};

template <MatrixConcept X, size_t C = std::min(X::rows, X::columns)>
struct Diagonal : MatrixView<X>, VectorStorageType<C, typename X::value_type> {
  using MatrixView<X>::MatrixView;
  using result_type = Vector<C, typename X::value_type>;

  constexpr auto& operator[](size_t const i) {
    return this->matrix[i, i];
  }
  constexpr auto operator[](size_t const i) const {
    return std::as_const(this->matrix)[i, i];
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
  using result_type = Vector<S, typename X::value_type>;
  constexpr auto& operator[](size_t const i) {
    return this->matrix[this->index, i];
  }
  constexpr auto operator[](size_t const i) const {
    return std::as_const(this->matrix)[this->index, i];
  }
};

template <MatrixConcept X, size_t S>
struct Column : RowColumnView<X, S>,
                VectorStorageType<X::rows, typename X::value_type> {
  using RowColumnView<X, S>::RowColumnView;
  using result_type = Vector<S, typename X::value_type>;
  constexpr auto& operator[](size_t const i) {
    return this->matrix[i, this->index];
  }
  constexpr auto operator[](size_t const i) const {
    return std::as_const(this->matrix)[i, this->index];
  }
};

template <size_t N, size_t M, typename T,
          template <size_t, size_t, typename C> typename Storage>
  requires(N > 0 && M > 0)
struct Matrix : Storage<N, M, T> {
  using Storage<N, M, T>::Storage;
  using result_type = Matrix;

  template <size_t P, size_t Q>
  using resized = Matrix<P, Q, T, Storage>;
  template <typename V>
  using retyped = Matrix<N, M, V, Storage>;
  template <MatrixConcept O>
  explicit constexpr Matrix(
      O const& other, [[maybe_unused]] ConstructorTag tag = ConstructorTag{}) {
    this->for_each_element([this, &other](size_t const i, size_t const j) {
      (*this)[i, j] = other[i, j];
    });
  }

  template <MatrixConcept O>
    requires(O::rows == N && O::columns == M)
  constexpr Matrix& operator+=(O const& other) {
    this->for_each_element([this, &other](size_t const i, size_t const j) {
      (*this)[i, j] += other[i, j];
    });
    return *this;
  }

  template <MatrixConcept O>
    requires(O::rows == N && O::columns == M)
  constexpr Matrix& operator-=(O const& other) {
    this->for_each_element([this, &other](size_t const i, size_t const j) {
      (*this)[i, j] -= other[i, j];
    });
    return *this;
  }

  constexpr Matrix& operator*=(ScalarConcept auto const value) {
    this->for_each_element([this, value](size_t const i, size_t const j) {
      (*this)[i, j] *= value;
    });
    return *this;
  }

  template <typename Self>
  constexpr auto& transpose(this Self&& self) {
    static auto result = Transpose<noref<Self>>{self};
    return result;
  }

  template <typename Self>
  constexpr auto& diagonal(this Self&& self) {
    static auto result = Diagonal<noref<Self>>{self};
    return result;
  }

  template <typename Self>
  constexpr auto& row(this Self&& self, size_t const i) {
    static auto result = Row<noref<Self>, self.rows>{self, i};
    return result;
  }

  template <typename Self>
  constexpr auto& column(this Self&& self, size_t const i) {
    static auto result = Column<noref<Self>, self.columns>{self, i};
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
constexpr bool equals(X1 const& x1, X2 const& x2, std::index_sequence<Is...>) {
  return (... &&
          (RowAt<Is>::equals(x1, x2, std::make_index_sequence<X1::columns>{})));
}

template <MatrixConcept X1, MatrixConcept X2>
  requires(X1::rows == X2::rows && X1::columns == X2::columns)
constexpr bool operator==(X1 const& x1, X2 const& x2) {
  return equals(x1, x2, std::make_index_sequence<X1::rows>{});
}

template <MatrixConcept X1, MatrixConcept X2, typename value_type, size_t... Is>
constexpr bool close(X1 const& x1, X2 const& x2, value_type threshold,
                     std::index_sequence<Is...>) {
  return (... && (RowAt<Is>::close(x1, x2, threshold,
                                   std::make_index_sequence<X1::columns>{})));
}

template <MatrixConcept X1, MatrixConcept X2>
  requires(X1::rows == X2::rows && X1::columns == X2::columns)
constexpr bool all_close(
    X1 const& x1, X2 const& x2,
    typename X1::value_type threshold =
        1000 *
        std::numeric_limits<std::common_type_t<
            typename X1::value_type, typename X2::value_type>>::epsilon()) {
  return close(x1, x2, threshold, std::make_index_sequence<X1::rows>{});
}

template <MatrixConcept X1, MatrixConcept X2>
  requires(noref<X1>::rows == noref<X2>::rows &&
           noref<X1>::columns == noref<X2>::columns)
constexpr auto operator+(X1&& x1, X2&& x2) {
  return MatrixAddition<X1, X2>{std::forward<X1>(x1), std::forward<X2>(x2)};
}

template <MatrixConcept X1, MatrixConcept X2>
  requires(noref<X1>::rows == noref<X2>::rows &&
           noref<X1>::columns == noref<X2>::columns)
constexpr auto operator-(X1&& x1, X2&& x2) {
  return MatrixSubtraction<X1, X2>{std::forward<X1>(x1), std::forward<X2>(x2)};
}

template <MatrixConcept X, ScalarConcept S>
constexpr auto operator*(X&& x, S const value) {
  return MatrixScalarMultiplication<X, S>{std::forward<X>(x), value};
}

template <MatrixConcept X, ScalarConcept S>
constexpr auto operator*(S const value, X&& x) {
  return std::forward<X>(x) * value;
}

template <MatrixConcept X, VectorConcept V>
  requires(noref<X>::columns == noref<V>::components)
constexpr auto operator*(X&& x, V&& v) {
  return MatrixVectorMultiplication<X, V, false>{std::forward<X>(x),
                                                 std::forward<V>(v)};
}

template <MatrixConcept X, VectorConcept V>
  requires(noref<X>::rows == noref<V>::components)
constexpr auto operator*(V&& v, X&& x) {
  return MatrixVectorMultiplication<X, V, true>{std::forward<X>(x),
                                                std::forward<V>(v)};
}

template <MatrixConcept X1, MatrixConcept X2>
  requires(noref<X1>::columns == noref<X2>::rows)
constexpr auto operator*(X1&& x1, X2&& x2) {
  return MatrixMultiplication<X1, X2>{std::forward<X1>(x1),
                                      std::forward<X2>(x2)};
}

#ifdef HAVE_BLAS
template <size_t N, size_t K, size_t M>
constexpr auto blas_mul(GenericStorage<N, K, double> const& x1,
                        GenericStorage<K, M, double> const& x2) {
  auto result = x1.blas_mul(x2);
  return result;
}
#endif

template <size_t MinSize = 16>
  requires(MinSize > 0)
struct Strassen {
  template <MatrixConcept X1, MatrixConcept X2>
    requires(noref<X1>::columns == noref<X2>::rows)
  static constexpr auto operator()(X1&& x1, X2&& x2) {
    if constexpr (noref<X1>::rows <= MinSize || noref<X1>::columns <= MinSize ||
                  noref<X2>::columns <= MinSize) {
      return (std::forward<X1>(x1) * std::forward<X2>(x2))();
    } else {
      static constexpr size_t N =
          std::max((noref<X1>::rows + 1) / 2, (noref<X1>::columns + 1) / 2);
      using result_type =
          Matrix<noref<X1>::rows, noref<X2>::columns,
                 std::common_type_t<typename noref<X1>::value_type,
                                    typename noref<X2>::value_type>>;
      result_type result;
      auto a11 = SubMatrix<X1, 0, 0, N, N>{std::forward<X1>(x1)};
      auto a12 = SubMatrix<X1, 0, N, N, N>{std::forward<X1>(x1)};
      auto a21 = SubMatrix<X1, N, 0, N, N>{std::forward<X1>(x1)};
      auto a22 = SubMatrix<X1, N, N, N, N>{std::forward<X1>(x1)};
      auto b11 = SubMatrix<X2, 0, 0, N, N>{std::forward<X2>(x2)};
      auto b12 = SubMatrix<X2, 0, N, N, N>{std::forward<X2>(x2)};
      auto b21 = SubMatrix<X2, N, 0, N, N>{std::forward<X2>(x2)};
      auto b22 = SubMatrix<X2, N, N, N, N>{std::forward<X2>(x2)};
      auto r11 = SubMatrix<result_type&, 0, 0, N, N>{result};
      auto r12 = SubMatrix<result_type&, 0, N, N, N>{result};
      auto r21 = SubMatrix<result_type&, N, 0, N, N>{result};
      auto r22 = SubMatrix<result_type&, N, N, N, N>{result};
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
    requires(noref<X1>::columns == noref<X2>::rows)
  static constexpr auto operator()(X1&& x1, X2&& x2) {
    if constexpr (noref<X1>::rows <= MinSize || noref<X1>::columns <= MinSize ||
                  noref<X2>::columns <= MinSize) {
      return (std::forward<X1>(x1) * std::forward<X2>(x2))();
    } else {
      static constexpr size_t N =
          std::max((noref<X1>::rows + 1) / 2, (noref<X1>::columns + 1) / 2);
      using result_type =
          Matrix<noref<X1>::rows, noref<X2>::columns,
                 std::common_type_t<typename noref<X1>::value_type,
                                    typename noref<X2>::value_type>>;
      result_type result;
      auto a11 = SubMatrix<X1, 0, 0, N, N>{std::forward<X1>(x1)};
      auto a12 = SubMatrix<X1, 0, N, N, N>{std::forward<X1>(x1)};
      auto a21 = SubMatrix<X1, N, 0, N, N>{std::forward<X1>(x1)};
      auto a22 = SubMatrix<X1, N, N, N, N>{std::forward<X1>(x1)};
      auto b11 = SubMatrix<X2, 0, 0, N, N>{std::forward<X2>(x2)};
      auto b12 = SubMatrix<X2, 0, N, N, N>{std::forward<X2>(x2)};
      auto b21 = SubMatrix<X2, N, 0, N, N>{std::forward<X2>(x2)};
      auto b22 = SubMatrix<X2, N, N, N, N>{std::forward<X2>(x2)};
      auto r11 = SubMatrix<result_type&, 0, 0, N, N>{result};
      auto r12 = SubMatrix<result_type&, 0, N, N, N>{result};
      auto r21 = SubMatrix<result_type&, N, 0, N, N>{result};
      auto r22 = SubMatrix<result_type&, N, N, N, N>{result};
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
