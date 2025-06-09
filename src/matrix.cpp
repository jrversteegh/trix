#include <array>
#include <cassert>
#include <iostream>
#include <functional>
#include <initializer_list>
#include <random>

namespace trix {

struct MatrixConcept {
};

struct StorageConcept {
};

template <int N, int M, typename T=double>
struct ArrayStorage: public StorageConcept {
  static constexpr size_t row_count = N;
  static constexpr size_t column_count = M;
  static constexpr size_t element_count = row_count * column_count;
  template <std::convertible_to<T>... Values>
  constexpr ArrayStorage(Values... values): m_{values...} {};
  constexpr T operator[](const size_t i, const size_t j) const {
    size_t offset = i * M + j;
    assert(offset < element_count);
    return m_[offset];
  }
  constexpr T& operator[](const size_t i, const size_t j) {
    size_t offset = i * M + j;
    assert(offset < element_count);
    return m_[offset];
  }
  constexpr size_t size() const {
    return element_count;
  }
private:
  std::array<T, element_count> m_;
};

template <int N, int M, typename T=double, template<int P, int Q, typename C=double> typename Storage=ArrayStorage>
struct Matrix: public MatrixConcept, public Storage<N, M, T> {
  using Storage<N, M, T>::Storage;

  template <typename F>
  requires std::same_as<std::invoke_result_t<F, size_t, size_t>, bool>
  constexpr bool for_each_element_while_true(F fun) const {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        if (!fun(i, j))
          return false;
      }
    }
    return true;
  }

  template <typename F>
  requires std::same_as<std::invoke_result_t<F, size_t, size_t>, void>
  constexpr void for_each_element(F fun) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        fun(i, j);
      }
    }
  }

  constexpr Matrix& operator+=(Matrix const& other) {
    for_each_element([this, &other](size_t i, size_t j) { (*this)[i, j] += other[i, j]; });
    return *this;
  }

  constexpr Matrix& operator-=(Matrix const& other) {
    for_each_element([this, &other](size_t i, size_t j) { (*this)[i, j] -= other[i, j]; });
    return *this;
  }

  constexpr Matrix& operator*=(T const value) {
    for_each_element([this, value](size_t i, size_t j) { (*this)[i, j] *= value; });
    return *this;
  }

  constexpr bool operator==(Matrix const& other) const {
    return for_each_element_while_true([this, &other](size_t i, size_t j) -> bool { return (*this)[i, j] == other[i, j]; });
  }
};

template <int N, int M>
constexpr auto operator+(Matrix<N, M> const& m1, Matrix<N, M> const& m2) {
  Matrix<N, M> result{m1};
  result += m2;
  return result;
}

template <int N, int M>
constexpr auto operator-(Matrix<N, M> const& m1, Matrix<N, M> const& m2) {
  Matrix<N, M> result{m1};
  result -= m2;
  return result;
}

template <int N, int M, typename T=double>
constexpr auto operator*(Matrix<N, M> const& m, T const value) {
  Matrix<N, M> result{m};
  result *= value;
  return result;
}

template <int N, int M, typename T=double>
constexpr auto operator*(T const value, Matrix<N, M> const& m) {
  return m * value;
}

template <int N, int M, int P>
constexpr auto operator*(Matrix<N, M> const& m1, Matrix<M, P> const& m2) {
  Matrix<N, P> result;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < P; ++j) {
      for (int k = 0; k < M; ++k) {
        result[i, j] += m1[i, k] * m2[k, j];
      }
    }
  }
  return result;
}

template <int N, int M>
std::ostream& operator<<(std::ostream& out, Matrix<N, M> const& m) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      out << " " << m[i, j] << ",";
    }
    out << std::endl;
  }
  return out;
}

}  // namespace trix
