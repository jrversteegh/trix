#ifndef TRIX_VECTOR_H__
#define TRIX_VECTOR_H__

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


template <typename V>
concept VectorConcept = requires(V const v, size_t i) {
  typename V::value_type;
  { v.operator[](i) } -> std::same_as<typename V::value_type>;
  { v.components } -> std::convertible_to<size_t>;
};

template <typename V>
concept MutableVectorConcept = VectorConcept<V> && requires(V v, size_t i) {
  { v.operator[](i) } -> std::same_as<typename V::value_type&>;
};


struct VectorType {};

template <size_t N, typename T>
struct VectorStorageType {
  static constexpr size_t components = N;
  using value_type = T;
};


template <typename STORAGE, size_t N, size_t SIZE, typename T = Number>
struct VectorArrayStorage : VectorStorageType<N,  T> {
  static constexpr size_t elements = SIZE;
  template <std::convertible_to<T>... Values>
  constexpr VectorArrayStorage(Values... values) : m_{values...} {};
  constexpr VectorArrayStorage(VectorArrayStorage const&) = default;
  constexpr VectorArrayStorage(VectorArrayStorage&&) = default;
  constexpr T operator[](const size_t i) const {
    return m_[check_and_get_offset_(i)];
  }
  constexpr T &operator[](const size_t i, const size_t j) {
    return m_[check_and_get_offset_(i)];
  }
  constexpr size_t size() const { return elements; }

private:
  std::array<T, elements> m_{};
  static constexpr size_t check_and_get_offset_(const size_t i) {
    size_t offset = STORAGE::get_offset(i);
    assert(offset < elements);
    return offset;
  }
};

template <size_t N, typename T = Number>
struct VectorGenericStorage : VectorArrayStorage<VectorGenericStorage<N, T>, N, N, T> {
  using VectorArrayStorage<VectorGenericStorage<N, T>, N, N, T>::VectorArrayStorage;
  static constexpr size_t get_offset(const size_t i) {
    return i;
  }
};


template <size_t N, typename T = Number,
          template <size_t, typename C> typename Storage =
              VectorGenericStorage>
requires (N > 0)
struct Vector : Storage<N, T>, VectorType {
  using Storage<N, T>::Storage;
  constexpr Vector(Vector const&) = default;
  constexpr Vector(Vector&&) = default;
  template <VectorConcept OTHER>
    requires (OTHER::components >= Vector::components)
  constexpr Vector(OTHER const& other): Storage<OTHER::components, typename OTHER::value_type>{} {
    this->for_each_element(
        [this, &other](size_t i) { (*this)[i] = other[i]; });
  }

  template <VectorConcept OTHER>
  requires (OTHER::components == N)
  constexpr Vector &operator+=(OTHER const &other) {
    this->for_each_element(
        [this, &other](size_t i) { (*this)[i] += other[i]; });
    return *this;
  }

  template <VectorConcept OTHER>
  requires (OTHER::components == N)
  constexpr Vector &operator-=(OTHER const &other) {
    this->for_each_element(
        [this, &other](size_t i) { (*this)[i] -= other[i]; });
    return *this;
  }

  constexpr Vector &operator*=(ScalarConcept auto const value) {
    this->for_each_element(
        [this, value](size_t i) { (*this)[i] *= value; });
    return *this;
  }

  constexpr bool operator==(Vector const &other) const {
    return this->for_each_element_while_true(
        [this, &other](size_t i) -> bool {
          return (*this)[i] == other[i];
        });
  }

  template <VectorConcept OTHER>
  requires (OTHER::components == N)
  constexpr bool operator==(OTHER const &other) const {
    for (size_t i = 0; i < N; ++i) {
      if ((*this)[i] != other[i])
        return false;
    }
    return true;
  }
  static_assert(VectorConcept<Vector>, "Excepted Vector to satisfy VectorConcept");

};


template <VectorConcept V1, VectorConcept V2>
requires (V1::components == V2::components)
constexpr auto operator+(V1 const &v1, V2 const &v2) {
  Vector<V1::components, std::common_type_t<typename V1::value_type, typename V2::value_type>> result{v1};
  result += v2;
  return result;
}

template <VectorConcept V1, VectorConcept V2>
requires (V1::components == V2::components)
constexpr auto operator-(V1 const &v1, V2 const &v2) {
  Vector<V1::components, std::common_type_t<typename V1::value_type, typename V2::value_type>> result{v1};
  result -= v2;
  return result;
}


template <size_t N, typename T, template <size_t, typename> typename S, ScalarConcept V>
requires MutableVectorConcept<Vector<N, T, S>>
constexpr auto operator*(Vector<N, T, S> const &v, V const value) {
  Vector<N, T, S> result{v};
  result *= value;
  return result;
}

template <VectorConcept VEC, ScalarConcept V>
constexpr auto operator*(VEC const &v, VEC const value) {
  Vector<VEC::components, typename V::value_type> result{v};
  result *= value;
  return result;
}


template <VectorConcept V1, VectorConcept V2>
requires (V1::components == V2::components)
constexpr auto operator*(V1 const &v1, V2 const &v2) {
  std::common_type_t<typename V1::value_type, typename V2::value_type> result{v1[0] * v2[0]};
  for (size_t i = 1; i < V1::components; ++i) {
    result[i] += v1[i] * v2[i];
  }
  return result;
}

template <VectorConcept V>
std::ostream &operator<<(std::ostream &out, V const &v) {
  for (size_t i = 0; i < V::components; ++i) {
    out << " " << v[i] << ",";
  }
  return out;
}

} // namespace trix

#endif
