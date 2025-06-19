#include <array>
#include <cassert>
#include <functional>
#include <ostream>
#include <sstream>
#include <random>
#include <type_traits>

#include "config.h"

namespace trix {

struct MatrixType {};

struct StorageType {};

struct SymmetricType {};

template <typename T>
concept Scalar = std::is_integral_v<T> || std::is_floating_point_v<T>;

template <typename STORAGE, size_t N, size_t M, size_t SIZE, typename T = Number>
struct ArrayStorage : public StorageType {
  static constexpr size_t row_count = N;
  static constexpr size_t column_count = M;
  static constexpr size_t element_count = SIZE;
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
  constexpr size_t size() const { return element_count; }

private:
  std::array<T, element_count> m_{};
  static constexpr size_t check_and_get_offset_(const size_t i, const size_t j) {
    size_t offset = STORAGE::get_offset(i, j);
    assert(offset < element_count);
    return offset;
  }
};

template <size_t N, size_t M, typename T = Number>
struct GenericStorage : public ArrayStorage<GenericStorage<N, M, T>, N, M, N * M, T> {
  using ArrayStorage<GenericStorage<N, M, T>, N, M, N * M, T>::ArrayStorage;
  constexpr GenericStorage(GenericStorage const&) = default;
  constexpr GenericStorage(GenericStorage&&) = default;
  static constexpr size_t get_offset(const size_t i, const size_t j) {
    return i * M + j;
  }

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

template <size_t N, size_t M=N, typename T = Number>
requires (M == N)
struct SymmetricStorage : public ArrayStorage<SymmetricStorage<N, M, T>, N, M, N * (N + 1) / 2, T>, public SymmetricType {
  using ArrayStorage<SymmetricStorage<N, M, T>, N, M, N * (N + 1) / 2, T>::ArrayStorage;
  constexpr SymmetricStorage(SymmetricStorage const&) = default;
  constexpr SymmetricStorage(SymmetricStorage&&) = default;
  static constexpr size_t get_offset(const size_t i, const size_t j) {
    return i > j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
  }

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

template <size_t N, size_t M=N, typename T = Number,
          template <size_t, size_t, typename C> typename Storage =
              GenericStorage>
requires (N > 0 && M > 0)
struct Matrix : public MatrixType, public Storage<N, M, T> {
  using Storage<N, M, T>::Storage;
  constexpr Matrix(Matrix const&) = default;
  constexpr Matrix(Matrix&&) = default;
  template <typename TO, template<size_t, size_t, typename> typename SO>
  constexpr Matrix(Matrix<N, M, TO, SO> const& other): Storage<N, M, T>{} {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] = other[i, j]; });
  }

  template <typename TO, template<size_t, size_t, typename> typename SO>
  constexpr Matrix &operator+=(Matrix<N, M, TO, SO> const &other) {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] += other[i, j]; });
    return *this;
  }

  template <typename TO, template<size_t, size_t, typename> typename SO>
  constexpr Matrix &operator-=(Matrix<N, M, TO, SO> const &other) {
    this->for_each_element(
        [this, &other](size_t i, size_t j) { (*this)[i, j] -= other[i, j]; });
    return *this;
  }

  constexpr Matrix &operator*=(Scalar auto const value) {
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

  template <typename TO, template<size_t, size_t, typename> typename SO>
  constexpr bool operator==(Matrix<N, M, TO, SO> const &other) const {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < M; ++j) {
        if ((*this)[i, j] != other[i, j])
          return false;
      }
    }
    return true;
  }

};

template <size_t N, typename T = Number>
using SymmetricMatrix = Matrix<N, N, T, SymmetricStorage>;

template <size_t N, size_t M, typename T1, typename T2, template <size_t, size_t, typename> typename S1, template <size_t, size_t, typename> typename S2>
constexpr auto operator+(Matrix<N, M, T1, S1> const &m1, Matrix<N, M, T2, S2> const &m2) {
  Matrix<N, M, T1> result{m1};
  result += m2;
  return result;
}

template <size_t N, typename T>
constexpr auto operator+(SymmetricMatrix<N, T> const &m1, SymmetricMatrix<N, T> const &m2) {
  SymmetricMatrix<N, T> result{m1};
  result += m2;
  return result;
}

template <size_t N, size_t M, typename T1, typename T2, template <size_t, size_t, typename> typename S1, template <size_t, size_t, typename> typename S2>
constexpr auto operator-(Matrix<N, M, T1, S1> const &m1, Matrix<N, M, T2, S2> const &m2) {
  Matrix<N, M, T1> result{m1};
  result -= m2;
  return result;
}

template <size_t N, typename T>
constexpr auto operator-(SymmetricMatrix<N, T> const &m1, SymmetricMatrix<N, T> const &m2) {
  SymmetricMatrix<N, T> result{m1};
  result -= m2;
  return result;
}

template <size_t N, size_t M, typename T, template <size_t, size_t, typename> typename S, Scalar V>
constexpr auto operator*(Matrix<N, M, T, S> const &m, V const value) {
  Matrix<N, M, T, S> result{m};
  result *= value;
  return result;
}

template <size_t N, size_t M, typename T, template <size_t, size_t, typename> typename S, Scalar V>
constexpr auto operator*(V const value, Matrix<N, M, T, S> const &m) {
  return m * value;
}

template <size_t N, size_t M, size_t P, typename T, template <size_t, size_t, typename> typename S1, template <size_t, size_t, typename> typename S2>
constexpr auto operator*(Matrix<N, M, T, S1> const &m1, Matrix<M, P, T, S2> const &m2) {
  Matrix<N, P, T> result;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < P; ++j) {
      result[i, j] = m1[i, 0] * m2[0, j];
      for (size_t k = 1; k < M; ++k) {
        result[i, j] += m1[i, k] * m2[k, j];
      }
    }
  }
  return result;
}

template <size_t N, typename T>
constexpr auto operator*(SymmetricMatrix<N, T> const &m1, SymmetricMatrix<N, T> const &m2) {
  SymmetricMatrix<N, T> result;
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

template <std::convertible_to<MatrixType> M>
std::ostream &operator<<(std::ostream &out, M const &m) {
  for (size_t i = 0; i < M::row_count; ++i) {
    for (size_t j = 0; j < M::column_count; ++j) {
      out << " " << m[i, j] << ",";
    }
    out << std::endl;
  }
  return out;
}

} // namespace trix
