#ifndef TRIX_MATRIX_H__
#define TRIX_MATRIX_H__

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <ostream>
#include <random>
#include <sstream>
#include <type_traits>

#include "config.h"
#include "vector.h"

namespace trix {


template <typename M>
concept MatrixConcept = requires(M const m, size_t i, size_t j) {
  typename M::value_type;
  { m.operator[](i, j) } -> std::same_as<typename M::value_type>;
  { m.rows } -> std::convertible_to<size_t>;
  { m.columns } -> std::convertible_to<size_t>;
};

template <typename M>
concept MutableMatrixConcept = MatrixConcept<M> && requires(M m, size_t i, size_t j) {
  { m.operator[](i, j) } -> std::same_as<typename M::value_type&>;
};

struct MatrixType {};

template <size_t N, size_t M, typename T>
struct StorageType {
  static constexpr size_t rows = N;
  static constexpr size_t columns = M;
  using value_type = T;
};

struct RectangularBase {};

template <size_t N, size_t M>
struct RectangularType: RectangularBase {
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

struct SymmetricBase: RectangularBase {};

template <size_t N>
struct SymmetricType: SymmetricBase {
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

struct DiagonalBase: SymmetricBase {};

template <size_t N>
struct DiagonalType: DiagonalBase {
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


template <typename STORAGE, size_t N, size_t M, size_t SIZE, typename T = Number>
struct ArrayStorage : StorageType<N, M, T> {
  static constexpr size_t elements = SIZE;
  template <std::convertible_to<T>... Values>
  constexpr ArrayStorage(Values... values) : m_{values...} {};
  constexpr ArrayStorage(ArrayStorage const&) = default;
  constexpr ArrayStorage(ArrayStorage&&) = default;
  constexpr T operator[](const size_t i, const size_t j) const {
    return m_[check_and_get_offset_(i, j)];
  }
  constexpr T &operator[](const size_t i, const size_t j) {
    return m_[check_and_get_offset_(i, j)];
  }
  constexpr size_t size() const { return elements; }

private:
  std::array<T, elements> m_{};
  static constexpr size_t check_and_get_offset_(const size_t i, const size_t j) {
    size_t offset = STORAGE::get_offset(i, j);
    assert(offset < elements);
    return offset;
  }
};

template <size_t N, size_t M, typename T = Number>
struct GenericStorage : ArrayStorage<GenericStorage<N, M, T>, N, M, N * M, T>, RectangularType<N, M> {
  using ArrayStorage<GenericStorage<N, M, T>, N, M, N * M, T>::ArrayStorage;
  static constexpr size_t get_offset(const size_t i, const size_t j) {
    return i * M + j;
  }

};

template <size_t N, size_t M=N, typename T = Number>
requires (M == N)
struct SymmetricStorage : ArrayStorage<SymmetricStorage<N, M, T>, N, M, N * (N + 1) / 2, T>, SymmetricType<N> {
  using ArrayStorage<SymmetricStorage<N, M, T>, N, M, N * (N + 1) / 2, T>::ArrayStorage;
  static constexpr size_t get_offset(const size_t i, const size_t j) {
    return i > j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
  }

};

template <size_t N, size_t M=N, typename T = Number>
requires (M == N)
struct DiagonalStorage : ArrayStorage<DiagonalStorage<N, M, T>, N, M, N, T>, DiagonalType<N> {
  using Base =ArrayStorage<DiagonalStorage<N, M, T>, N, M, N, T>;
  using Base::Base;
  static constexpr size_t get_offset(const size_t i, [[maybe_unused]] const size_t j) {
    return i;
  }
  constexpr T operator[](const size_t i, const size_t j) const {
    return i == j ? Base::operator[](i, j) : 0;
  }
  constexpr T &operator[](const size_t i, const size_t j) {
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

template <size_t N, size_t M=N, typename T = Number>
requires (M == N)
struct IdentityStorage : DiagonalType<N>, StorageType<N, N, T> {
  static constexpr size_t rows = N;
  static constexpr size_t columns = N;
  constexpr T operator[](const size_t i, const size_t j) const {
    return i == j ? 1 : 0;
  }
};

template <size_t N, size_t M=N, typename T = Number,
          template <size_t, size_t, typename C> typename Storage =
              GenericStorage>
requires (N > 0 && M > 0)
struct Matrix : Storage<N, M, T>, MatrixType {
  using Storage<N, M, T>::Storage;
  constexpr Matrix(Matrix const&) = default;
  constexpr Matrix(Matrix&&) = default;
  template <MatrixConcept OTHER>
  constexpr Matrix(OTHER const& other): Storage<OTHER::rows, OTHER::columns, typename OTHER::value_type>{} {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] = other[i, j]; });
  }

  template <MatrixConcept OTHER>
  requires (OTHER::rows == N && OTHER::columns == M)
  constexpr Matrix &operator+=(OTHER const &other) {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] += other[i, j]; });
    return *this;
  }

  template <MatrixConcept OTHER>
  requires (OTHER::rows == N && OTHER::columns == M)
  constexpr Matrix &operator-=(OTHER const &other) {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] -= other[i, j]; });
    return *this;
  }

  constexpr Matrix &operator*=(ScalarConcept auto const value) {
    this->for_each_element(
        [this, value](size_t i, size_t j) { (*this)[i, j] *= value; });
    return *this;
  }

  constexpr bool operator==(Matrix const &other) const {
    return this->for_each_element_while_true(
        [this, &other](size_t i, size_t j) -> bool {
          return (*this)[i, j] == other[i, j];
        });
  }

  template <MatrixConcept OTHER>
  requires (OTHER::rows == N && OTHER::columns == M)
  constexpr bool operator==(OTHER const &other) const {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j) {
        if ((*this)[i, j] != other[i, j])
          return false;
      }
    }
    return true;
  }
  static_assert(MatrixConcept<Matrix>, "Excepted Matrix to satisfy MatrixConcept");

  struct Transpose {
    using value_type = T;
    static constexpr size_t rows = M;
    static constexpr size_t columns = N;
    constexpr Transpose(Matrix& matrix): matrix_(matrix) {}
    constexpr T operator[](const size_t i, const size_t j) const {
      return matrix_[j, i];
    }
    constexpr T &operator[](const size_t i, const size_t j) {
      return matrix_[j, i];
    }
  private:
    Matrix& matrix_;
  };
  static_assert(MatrixConcept<Transpose>, "Excepted Transpose to satisfy MatrixConcept");

  constexpr Transpose& transpose() {
    static Transpose result{*this};
    return result;
  }
};



template <size_t N, typename T = Number>
using SymmetricMatrix = Matrix<N, N, T, SymmetricStorage>;

template <size_t N, typename T = Number>
using DiagonalMatrix = Matrix<N, N, T, DiagonalStorage>;

template <size_t N, typename T = Number>
using IdentityMatrix = Matrix<N, N, T, IdentityStorage>;


template <MatrixConcept M1, MatrixConcept M2>
requires (M1::rows == M2::rows && M1::columns == M2::columns)
constexpr auto operator+(M1 const &m1, M2 const &m2) {
  Matrix<M1::rows, M1::columns, std::common_type_t<typename M1::value_type, typename M2::value_type>> result{m1};
  result += m2;
  return result;
}

template <size_t N, typename T1, typename T2>
constexpr auto operator+(SymmetricMatrix<N, T1> const &m1, SymmetricMatrix<N, T2> const &m2) {
  SymmetricMatrix<N, std::common_type_t<T1, T2>> result{m1};
  result += m2;
  return result;
}

template <size_t N, typename T1, typename T2, template <size_t, size_t, typename> typename S1, template <size_t, size_t, typename> typename S2>
requires std::derived_from<Matrix<N, N, T1, S1>, DiagonalBase> && std::derived_from<Matrix<N, N, T2, S2>, DiagonalBase>
constexpr auto operator+(Matrix<N, N, T1, S1> const &m1, Matrix<N, N, T2, S2> const &m2) {
  DiagonalMatrix<N, std::common_type_t<T1, T2>> result{m1};
  result += m2;
  return result;
}

template <MatrixConcept M1, MatrixConcept M2>
requires (M1::rows == M2::rows && M1::columns == M2::columns)
constexpr auto operator-(M1 const &m1, M2 const &m2) {
  Matrix<M1::rows, M1::columns, std::common_type_t<typename M1::value_type, typename M2::value_type>> result{m1};
  result -= m2;
  return result;
}

template <size_t N, typename T1, typename T2>
constexpr auto operator-(SymmetricMatrix<N, T1> const &m1, SymmetricMatrix<N, T2> const &m2) {
  SymmetricMatrix<N, std::common_type_t<T1, T2>> result{m1};
  result -= m2;
  return result;
}

template <size_t N, typename T1, typename T2, template <size_t, size_t, typename> typename S1, template <size_t, size_t, typename> typename S2>
requires std::derived_from<Matrix<N, N, T1, S1>, DiagonalBase> && std::derived_from<Matrix<N, N, T2, S2>, DiagonalBase>
constexpr auto operator-(Matrix<N, N, T1, S1> const &m1, Matrix<N, N, T2, S2> const &m2) {
  DiagonalMatrix<N, std::common_type_t<T1, T2>> result{m1};
  result -= m2;
  return result;
}

template <size_t N, size_t M, typename T, template <size_t, size_t, typename> typename S, ScalarConcept V>
requires MutableMatrixConcept<Matrix<N, M, T, S>>
constexpr auto operator*(Matrix<N, M, T, S> const &m, V const value) {
  Matrix<N, M, T, S> result{m};
  result *= value;
  return result;
}

template <MatrixConcept M, ScalarConcept V>
requires std::derived_from<DiagonalBase, M>
constexpr auto operator*(M const &m, V const value) {
  DiagonalMatrix<M::rows, typename M::value_type> result{m};
  result *= value;
  return result;
}

template <MatrixConcept M, ScalarConcept V>
constexpr auto operator*(M const &m, V const value) {
  Matrix<M::rows, M::columns, typename M::value_type> result{m};
  result *= value;
  return result;
}

template <MatrixConcept M, ScalarConcept V>
constexpr auto operator*(V const value, M const &m) {
  return m * value;
}

template <MatrixConcept M1, MatrixConcept M2>
requires (M1::columns == M2::rows)
constexpr auto operator*(M1 const &m1, M2 const &m2) {
  Matrix<M1::rows, M2::columns, std::common_type_t<typename M1::value_type, typename M2::value_type>> result;
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
constexpr auto operator*(SymmetricMatrix<N, T1> const &m1, SymmetricMatrix<N, T2> const &m2) {
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
constexpr auto operator*(DiagonalMatrix<N, T1> const &m1, DiagonalMatrix<N, T2> const &m2) {
  DiagonalMatrix<N, std::common_type_t<T1, T2>> result;
  for (size_t i = 0; i < N; ++i) {
    result[i, i] = m1[i, i] * m2[i, i];
  }
  return result;
}

template <MatrixConcept M>
std::ostream &operator<<(std::ostream &out, M const &m) {
  for (size_t i = 0; i < M::rows; ++i) {
    for (size_t j = 0; j < M::columns; ++j) {
      out << " " << m[i, j] << ",";
    }
    out << std::endl;
  }
  return out;
}

} // namespace trix

#endif
