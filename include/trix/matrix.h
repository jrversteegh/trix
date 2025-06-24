#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <ostream>
#include <random>
#include <sstream>
#include <type_traits>

#include "config.h"

namespace trix {

using size_t = std::size_t;

template <typename M>
concept MatrixConcept = requires(M const m, size_t i, size_t j) {
  typename M::value_type;
  { m.operator[](i, j) } -> std::same_as<typename M::value_type>;
  { m.row_count } -> std::convertible_to<size_t>;
  { m.column_count } -> std::convertible_to<size_t>;
};

template <typename T>
concept ScalarConcept = std::is_integral_v<T> || std::is_floating_point_v<T>;

struct MatrixType {};

template <size_t N, size_t M, typename T>
struct StorageType {
  static constexpr size_t row_count = N;
  static constexpr size_t column_count = M;
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

struct SymmetricBase {};

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

struct DiagonalBase {};

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
struct GenericStorage : ArrayStorage<GenericStorage<N, M, T>, N, M, N * M, T>, RectangularType<N, M> {
  using ArrayStorage<GenericStorage<N, M, T>, N, M, N * M, T>::ArrayStorage;
  constexpr GenericStorage(GenericStorage const&) = default;
  constexpr GenericStorage(GenericStorage&&) = default;
  static constexpr size_t get_offset(const size_t i, const size_t j) {
    return i * M + j;
  }

};

template <size_t N, size_t M=N, typename T = Number>
requires (M == N)
struct SymmetricStorage : ArrayStorage<SymmetricStorage<N, M, T>, N, M, N * (N + 1) / 2, T>, SymmetricType<N> {
  using ArrayStorage<SymmetricStorage<N, M, T>, N, M, N * (N + 1) / 2, T>::ArrayStorage;
  constexpr SymmetricStorage(SymmetricStorage const&) = default;
  constexpr SymmetricStorage(SymmetricStorage&&) = default;
  static constexpr size_t get_offset(const size_t i, const size_t j) {
    return i > j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
  }

};

template <size_t N, size_t M=N, typename T = Number>
requires (M == N)
struct DiagonalStorage : ArrayStorage<DiagonalStorage<N, M, T>, N, M, N, T>, DiagonalType<N> {
  using Base =ArrayStorage<DiagonalStorage<N, M, T>, N, M, N, T>;
  //using ArrayStorage<DiagonalStorage<N, M, T>, N, M, N, T>::ArrayStorage;
  using Base::Base;
  constexpr DiagonalStorage(DiagonalStorage const&) = default;
  constexpr DiagonalStorage(DiagonalStorage&&) = default;
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
  static constexpr size_t row_count = N;
  static constexpr size_t column_count = N;
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
  static_assert(MatrixConcept<Matrix>, "Excepted Matrix to satisfy MatrixConcept");
};


template <size_t N, typename T = Number>
using SymmetricMatrix = Matrix<N, N, T, SymmetricStorage>;

template <size_t N, typename T = Number>
using DiagonalMatrix = Matrix<N, N, T, DiagonalStorage>;

template <size_t N, typename T = Number>
using IdentityMatrix = Matrix<N, N, T, IdentityStorage>;

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

template <size_t N, typename T1, typename T2, template <size_t, size_t, typename> typename S1, template <size_t, size_t, typename> typename S2>
requires std::derived_from<Matrix<N, N, T1, S1>, DiagonalBase> && std::derived_from<Matrix<N, N, T2, S2>, DiagonalBase>
constexpr auto operator+(Matrix<N, N, T1, S1> const &m1, Matrix<N, N, T2, S2> const &m2) {
  DiagonalMatrix<N, T1> result{m1};
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

template <size_t N, size_t M, typename T, template <size_t, size_t, typename> typename S, ScalarConcept V>
constexpr auto operator*(Matrix<N, M, T, S> const &m, V const value) {
  Matrix<N, M, T, S> result{m};
  result *= value;
  return result;
}

template <size_t N, typename T, ScalarConcept V>
constexpr auto operator*(IdentityMatrix<N, T> const &m, V const value) {
  DiagonalMatrix<N, T> result{m};
  result *= value;
  return result;
}

template <size_t N, size_t M, typename T, template <size_t, size_t, typename> typename S, ScalarConcept V>
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

template <MatrixConcept M>
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
