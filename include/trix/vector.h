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
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "config.h"

namespace trix {

template <typename V>
concept VectorConcept = requires(V const v, size_t i) {
  typename V::value_type;
  { v.operator[](i) } -> std::same_as<typename V::value_type>;
  { V::components } -> std::convertible_to<size_t>;
};

template <typename V>
concept MutableVectorConcept = VectorConcept<V> && requires(V v, size_t i) {
  { v.operator[](i) } -> std::same_as<typename V::value_type &>;
};

struct VectorType {};

template <size_t N, typename T> struct VectorStorageType {
  static constexpr size_t components = N;
  using value_type = T;
};

template <size_t N> struct FullVectorType {
  template <typename F>
    requires std::same_as<std::invoke_result_t<F, size_t>, bool>
  constexpr bool for_each_element_while_true(F fun) const {
    for (size_t i = 0; i < N; ++i) {
      if (!fun(i))
        return false;
    }
    return true;
  }

  template <typename F>
    requires std::same_as<std::invoke_result_t<F, size_t>, void>
  constexpr void for_each_element(F fun) {
    for (size_t i = 0; i < N; ++i) {
      fun(i);
    }
  }
};

template <typename STORAGE, size_t N, size_t SIZE, typename T = Number>
struct VectorArrayStorage : VectorStorageType<N, T>, FullVectorType<N> {
  static constexpr size_t elements = SIZE;
  template <std::convertible_to<T>... Values>
  constexpr VectorArrayStorage(Values &&...values)
      : a_{std::forward<Values>(values)...} {};
  explicit constexpr VectorArrayStorage(std::array<T, elements> &&array)
      : a_{std::forward(array)} {};
  constexpr T operator[](const size_t i) const {
    return a_[check_and_get_offset_(i)];
  }
  constexpr T &operator[](const size_t i) {
    return a_[check_and_get_offset_(i)];
  }
  constexpr size_t size() const { return elements; }
  std::string str() const { return fmt::format(fmtstr, fmt::join(a_, ", ")); }

private:
  std::array<T, elements> a_{};
  static constexpr size_t check_and_get_offset_(const size_t i) {
    size_t offset = STORAGE::get_offset(i);
    assert(offset < elements);
    return offset;
  }
};

template <size_t N, typename T = Number>
struct VectorGenericStorage
    : VectorArrayStorage<VectorGenericStorage<N, T>, N, N, T> {
  using VectorArrayStorage<VectorGenericStorage<N, T>, N, N,
                           T>::VectorArrayStorage;
  static constexpr size_t get_offset(const size_t i) { return i; }
};

template <size_t N, typename T = Number,
          template <size_t, typename C> typename Storage = VectorGenericStorage>
  requires(N > 0)
struct Vector : Storage<N, T>, VectorType {
  using Storage<N, T>::Storage;
  template <VectorConcept OTHER>
    requires(OTHER::components >= Vector::components)
  explicit constexpr Vector(OTHER const &other)
      : Storage<OTHER::components, typename OTHER::value_type>{} {
    this->for_each_element([this, &other](size_t i) { (*this)[i] = other[i]; });
  }

  template <VectorConcept OTHER>
    requires(OTHER::components == N)
  constexpr Vector &operator+=(OTHER const &other) {
    this->for_each_element(
        [this, &other](size_t i) { (*this)[i] += other[i]; });
    return *this;
  }

  template <VectorConcept OTHER>
    requires(OTHER::components == N)
  constexpr Vector &operator-=(OTHER const &other) {
    this->for_each_element(
        [this, &other](size_t i) { (*this)[i] -= other[i]; });
    return *this;
  }

  constexpr Vector &operator*=(ScalarConcept auto const value) {
    this->for_each_element([this, value](size_t i) { (*this)[i] *= value; });
    return *this;
  }

  constexpr Vector operator-() const { return Vector{} - *this; }

  constexpr bool operator==(Vector const &other) const {
    return this->for_each_element_while_true(
        [this, &other](size_t i) -> bool { return (*this)[i] == other[i]; });
  }

  template <VectorConcept OTHER>
    requires(OTHER::components == N)
  constexpr bool operator==(OTHER const &other) const {
    for (size_t i = 0; i < N; ++i) {
      if ((*this)[i] != other[i])
        return false;
    }
    return true;
  }

  static_assert(VectorConcept<Vector>,
                "Excepted Vector to satisfy VectorConcept");
};

template <typename C, typename... Cs>
auto vector(C &&first, Cs &&...components) {
  return Vector<sizeof...(Cs) + 1, C>{std::forward<C>(first),
                                      std::forward<Cs>(components)...};
}

template <VectorConcept V1, VectorConcept V2>
  requires(V1::components == V2::components)
constexpr auto operator+(V1 const &v1, V2 const &v2) {
  Vector<V1::components,
         std::common_type_t<typename V1::value_type, typename V2::value_type>>
      result{v1};
  result += v2;
  return result;
}

template <VectorConcept V1, VectorConcept V2>
  requires(V1::components == V2::components)
constexpr auto operator-(V1 const &v1, V2 const &v2) {
  Vector<V1::components,
         std::common_type_t<typename V1::value_type, typename V2::value_type>>
      result{v1};
  result -= v2;
  return result;
}

template <size_t N, typename T, template <size_t, typename> typename S,
          ScalarConcept V>
  requires MutableVectorConcept<Vector<N, T, S>>
constexpr auto operator*(Vector<N, T, S> const &v, V const value) {
  Vector<N, T, S> result{v};
  result *= value;
  return result;
}

template <VectorConcept V, ScalarConcept S>
constexpr auto operator*(V const &v, S const value) {
  Vector<V::components, typename V::value_type> result{v};
  result *= value;
  return result;
}

template <VectorConcept V, ScalarConcept S>
constexpr auto operator*(S const value, V const &v) {
  return v * value;
}

template <VectorConcept V1, VectorConcept V2>
  requires(V1::components == V2::components)
constexpr auto operator*(V1 const &v1, V2 const &v2) {
  std::common_type_t<typename V1::value_type, typename V2::value_type> result{
      v1[0] * v2[0]};
  for (size_t i = 1; i < V1::components; ++i) {
    result += v1[i] * v2[i];
  }
  return result;
}

template <VectorConcept V1, VectorConcept V2>
  requires(V1::components == 3 && V2::components == 3)
constexpr auto cross(V1 const &v1, V2 const &v2) {
  return vector(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0]);
}

template <VectorConcept V>
std::ostream &operator<<(std::ostream &out, V const &v) {
  out << v.str();
  return out;
}

} // namespace trix

#endif
