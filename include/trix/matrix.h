#ifndef TRIX_MATRIX_H__
#define TRIX_MATRIX_H__

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <ostream>
#include <random>
#include <ranges>
#include <sstream>
#include <type_traits>

#include "config.h"
#include "vector.h"

namespace trix {

template <typename M>
concept MatrixConcept = requires(M const m, size_t i, size_t j) {
  typename M::value_type;
  { m.operator[](i, j) } -> std::same_as<typename M::value_type>;
  { M::rows } -> std::convertible_to<size_t>;
  { M::columns } -> std::convertible_to<size_t>;
};

template <typename M>
concept MutableMatrixConcept =
    MatrixConcept<M> && requires(M m, size_t i, size_t j) {
      { m.operator[](i, j) } -> std::same_as<typename M::value_type&>;
    };

struct MatrixType {};

template <size_t N, size_t M, typename T>
struct MatrixStorageType {
  static constexpr size_t rows = N;
  static constexpr size_t columns = M;
  using value_type = T;
};

struct RectangularBase {};

template <size_t N, size_t M>
struct RectangularType : RectangularBase {
  template <typename F>
    requires std::same_as<std::invoke_result_t<F, size_t, size_t>, bool>
  constexpr bool for_each_element_while_true(F fun) const {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j) {
        if (!fun(i, j))
          return false;
      }
    }
    return true;
  }

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
    requires std::same_as<std::invoke_result_t<F, size_t, size_t>, bool>
  constexpr bool for_each_element_while_true(F fun) const {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        if (!fun(i, j))
          return false;
      }
    }
    return true;
  }

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
    requires std::same_as<std::invoke_result_t<F, size_t, size_t>, bool>
  constexpr bool for_each_element_while_true(F fun) const {
    for (size_t i = 0; i < N; ++i) {
      if (!fun(i, i))
        return false;
    }
    return true;
  }

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
  template <std::input_iterator It, typename Ite>
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

private:
  std::array<T, elements> a_{};
  static constexpr size_t check_and_get_offset_(size_t const i,
                                                size_t const j) {
    size_t offset = STORAGE::get_offset(i, j);
    assert(offset < elements);
    return offset;
  }
};

template <size_t N, size_t M, typename T = Number>
struct GenericStorage
    : MatrixArrayStorage<GenericStorage<N, M, T>, N, M, N * M, T>,
      RectangularType<N, M> {
  using MatrixArrayStorage<GenericStorage<N, M, T>, N, M, N * M,
                           T>::MatrixArrayStorage;
  static constexpr size_t get_offset(size_t const i, size_t const j) {
    return i * M + j;
  }
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
    requires std::same_as<std::invoke_result_t<F, size_t, size_t>, bool>
  constexpr bool for_each_element_while_true(F fun) const {
    for (size_t i = 0; i < N; ++i) {
      if (!fun(i, i))
        return false;
    }
    return true;
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
  static constexpr size_t rows = N;
  static constexpr size_t columns = N;
  constexpr T operator[](size_t const i, size_t const j) const {
    return i == j ? 1 : 0;
  }
};

template <size_t N, size_t M = N, typename T = Number,
          template <size_t, size_t, typename C> typename Storage =
              GenericStorage>
  requires(N > 0 && M > 0)
struct Matrix : Storage<N, M, T>, MatrixType {
  using Storage<N, M, T>::Storage;
  template <MatrixConcept OTHER>
  explicit constexpr Matrix(OTHER const& other)
      : Storage<OTHER::rows, OTHER::columns, typename OTHER::value_type>{} {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] = other[i, j]; });
  }

  template <MatrixConcept OTHER>
    requires(OTHER::rows == N && OTHER::columns == M)
  constexpr Matrix& operator+=(OTHER const& other) {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] += other[i, j]; });
    return *this;
  }

  template <MatrixConcept OTHER>
    requires(OTHER::rows == N && OTHER::columns == M)
  constexpr Matrix& operator-=(OTHER const& other) {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] -= other[i, j]; });
    return *this;
  }

  constexpr Matrix& operator*=(ScalarConcept auto const value) {
    this->for_each_element(
        [this, value](size_t i, size_t j) { (*this)[i, j] *= value; });
    return *this;
  }

  constexpr bool operator==(Matrix const& other) const {
    return this->for_each_element_while_true(
        [this, &other](size_t i, size_t j) -> bool {
          return (*this)[i, j] == other[i, j];
        });
  }

  template <MatrixConcept OTHER>
    requires(OTHER::rows == N && OTHER::columns == M)
  constexpr bool operator==(OTHER const& other) const {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j) {
        if ((*this)[i, j] != other[i, j])
          return false;
      }
    }
    return true;
  }
  static_assert(MatrixConcept<Matrix>,
                "Excepted Matrix to satisfy MatrixConcept");

  struct View {
    constexpr View(Matrix& m) : matrix(m) {};
    Matrix& matrix;
  };

  struct Transpose : View, MatrixStorageType<M, N, T> {
    using View::View;
    constexpr T operator[](size_t const i, size_t const j) const {
      return this->matrix[j, i];
    }
    constexpr T& operator[](size_t const i, size_t const j) {
      return this->matrix[j, i];
    }
  };

  static_assert(MatrixConcept<Transpose>,
                "Excepted Transpose to satisfy MatrixConcept");

  constexpr Transpose& transpose() {
    static Transpose result{*this};
    return result;
  }

  constexpr Transpose const& transpose() const {
    static Transpose result{*this};
    return result;
  }

  struct Diagonal : View, VectorStorageType<std::min(N, M), T> {
    using View::View;
    constexpr T operator[](size_t const i) const {
      return this->matrix[i, i];
    }
  };
  static_assert(VectorConcept<Diagonal>,
                "Excepted Diagonal to satisfy VectorConcept");

  constexpr Diagonal& diagonal() {
    static Diagonal result{*this};
    return result;
  }

  constexpr Diagonal const& diagonal() const {
    static Diagonal result{*this};
    return result;
  }

  template <size_t S>
  struct RowColumnView {
    constexpr RowColumnView(Matrix const& m, size_t i) : matrix(m), index(i) {
      assert(i < S);
    };
    Matrix const& matrix;
    size_t index;
  };

  struct Row : RowColumnView<N>, VectorStorageType<M, T> {
    using RowColumnView<N>::RowColumnView;
    constexpr T operator[](size_t const i) const {
      return this->matrix[this->index, i];
    }
  };
  static_assert(VectorConcept<Row>, "Excepted Row to satisfy VectorConcept");

  constexpr Row& row(size_t const i) const {
    static Row result{*this, i};
    return result;
  }

  struct Column : RowColumnView<M>, VectorStorageType<N, T> {
    using RowColumnView<M>::RowColumnView;
    constexpr T operator[](size_t const i) const {
      return this->matrix[i, this->index];
    }
  };
  static_assert(VectorConcept<Column>,
                "Excepted Column to satisfy VectorConcept");

  constexpr Column& column(size_t const i) const {
    static Column result{*this, i};
    return result;
  }
};

template <MatrixConcept M>
constexpr auto to_string(M const& m) {
  std::array<std::string, M::rows> lines{};
  for (size_t i = 0; i < M::rows; ++i) {
    std::vector<typename M::value_type> values{};
    for (size_t j = 0; j < M::columns; ++j) {
      values.push_back(m[i, j]);
    }
    lines[i] = fmt::format(trix_fmtstr, fmt::join(values, ", "));
  }
  return fmt::format("{}", fmt::join(lines, "\n"));
}

template <size_t N, typename T = Number>
using SymmetricMatrix = Matrix<N, N, T, SymmetricStorage>;

template <size_t N, typename T = Number>
using DiagonalMatrix = Matrix<N, N, T, DiagonalStorage>;

template <size_t N, typename T = Number>
using IdentityMatrix = Matrix<N, N, T, IdentityStorage>;

template <typename C, typename... Cs>
auto matrix(C&& first, Cs&&... components) {
  constexpr size_t S = static_cast<size_t>(sqrt(sizeof...(Cs) + 1));
  return Matrix<S, S, C>{std::forward<C>(first),
                         std::forward<Cs>(components)...};
}

template <MatrixConcept M1, MatrixConcept M2>
  requires(M1::rows == M2::rows && M1::columns == M2::columns)
constexpr auto operator+(M1 const& m1, M2 const& m2) {
  Matrix<M1::rows, M1::columns,
         std::common_type_t<typename M1::value_type, typename M2::value_type>>
      result{m1};
  result += m2;
  return result;
}

template <size_t N, typename T1, typename T2>
constexpr auto operator+(SymmetricMatrix<N, T1> const& m1,
                         SymmetricMatrix<N, T2> const& m2) {
  SymmetricMatrix<N, std::common_type_t<T1, T2>> result{m1};
  result += m2;
  return result;
}

template <size_t N, typename T1, typename T2,
          template <size_t, size_t, typename> typename S1,
          template <size_t, size_t, typename> typename S2>
  requires std::derived_from<Matrix<N, N, T1, S1>, DiagonalBase> &&
           std::derived_from<Matrix<N, N, T2, S2>, DiagonalBase>
constexpr auto operator+(Matrix<N, N, T1, S1> const& m1,
                         Matrix<N, N, T2, S2> const& m2) {
  DiagonalMatrix<N, std::common_type_t<T1, T2>> result{m1};
  result += m2;
  return result;
}

template <MatrixConcept M1, MatrixConcept M2>
  requires(M1::rows == M2::rows && M1::columns == M2::columns)
constexpr auto operator-(M1 const& m1, M2 const& m2) {
  Matrix<M1::rows, M1::columns,
         std::common_type_t<typename M1::value_type, typename M2::value_type>>
      result{m1};
  result -= m2;
  return result;
}

template <size_t N, typename T1, typename T2>
constexpr auto operator-(SymmetricMatrix<N, T1> const& m1,
                         SymmetricMatrix<N, T2> const& m2) {
  SymmetricMatrix<N, std::common_type_t<T1, T2>> result{m1};
  result -= m2;
  return result;
}

template <size_t N, typename T1, typename T2,
          template <size_t, size_t, typename> typename S1,
          template <size_t, size_t, typename> typename S2>
  requires std::derived_from<Matrix<N, N, T1, S1>, DiagonalBase> &&
           std::derived_from<Matrix<N, N, T2, S2>, DiagonalBase>
constexpr auto operator-(Matrix<N, N, T1, S1> const& m1,
                         Matrix<N, N, T2, S2> const& m2) {
  DiagonalMatrix<N, std::common_type_t<T1, T2>> result{m1};
  result -= m2;
  return result;
}

template <size_t N, size_t M, typename T,
          template <size_t, size_t, typename> typename S, ScalarConcept V>
  requires MutableMatrixConcept<Matrix<N, M, T, S>>
constexpr auto operator*(Matrix<N, M, T, S> const& m, V const value) {
  Matrix<N, M, T, S> result{m};
  result *= value;
  return result;
}

template <MatrixConcept M, ScalarConcept V>
  requires std::derived_from<DiagonalBase, M>
constexpr auto operator*(M const& m, V const value) {
  DiagonalMatrix<M::rows, typename M::value_type> result{m};
  result *= value;
  return result;
}

template <MatrixConcept M, ScalarConcept V>
constexpr auto operator*(M const& m, V const value) {
  Matrix<M::rows, M::columns, typename M::value_type> result{m};
  result *= value;
  return result;
}

template <MatrixConcept M, ScalarConcept V>
constexpr auto operator*(V const value, M const& m) {
  return m * value;
}

template <MatrixConcept M1, MatrixConcept M2>
  requires(M1::columns == M2::rows)
constexpr auto operator*(M1 const& m1, M2 const& m2) {
  Matrix<M1::rows, M2::columns,
         std::common_type_t<typename M1::value_type, typename M2::value_type>>
      result;
  for (size_t i = 0; i < M1::rows; ++i) {
    for (size_t j = 0; j < M2::columns; ++j) {
      result[i, j] = m1[i, 0] * m2[0, j];
      for (size_t k = 1; k < M1::columns; ++k) {
        result[i, j] += m1[i, k] * m2[k, j];
      }
    }
  }
  return result;
}

template <size_t N, typename T1, typename T2>
constexpr auto operator*(SymmetricMatrix<N, T1> const& m1,
                         SymmetricMatrix<N, T2> const& m2) {
  SymmetricMatrix<N, std::common_type_t<T1, T2>> result;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      result[i, j] = m1[i, 0] * m2[0, j];
      for (size_t k = 1; k < N; ++k) {
        result[i, j] += m1[i, k] * m2[k, j];
      }
    }
  }
  return result;
}

template <size_t N, typename T1, typename T2>
constexpr auto operator*(DiagonalMatrix<N, T1> const& m1,
                         DiagonalMatrix<N, T2> const& m2) {
  DiagonalMatrix<N, std::common_type_t<T1, T2>> result;
  for (size_t i = 0; i < N; ++i) {
    result[i, i] = m1[i, i] * m2[i, i];
  }
  return result;
}

template <MatrixConcept M, VectorConcept V,
          typename T = std::common_type_t<typename M::value_type,
                                          typename V::value_type>>
  requires(M::columns == V::components)
constexpr auto operator*(M const& m, V const& v) {
  Vector<M::rows, T> result;
  for (size_t i = 0; i < M::rows; ++i) {
    result[i] = m[i, 0] * v[0];
    for (size_t j = 1; j < V::components; ++j) {
      result[i] += m[i, j] * v[j];
    }
  }
  return result;
}

template <MatrixConcept M, VectorConcept V,
          typename T = std::common_type_t<typename M::value_type,
                                          typename V::value_type>>
  requires(M::rows == V::components)
constexpr auto operator*(V const& v, M const& m) {
  Vector<M::columns, T> result;
  for (size_t i = 0; i < M::columns; ++i) {
    result[i] = v[0] * m[0, i];
    for (size_t j = 1; j < V::components; ++j) {
      result[i] += v[j] * m[j, i];
    }
  }
  return result;
}

template <MatrixConcept M>
std::ostream& operator<<(std::ostream& out, M const& m) {
  out << to_string(m);
  return out;
}

} // namespace trix

#endif
