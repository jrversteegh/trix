#ifndef TRIX_VECTOR_H__
#define TRIX_VECTOR_H__

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <random>
#include <ranges>
#include <type_traits>
#include <utility>

#include "config.h"
#include "types.h"

namespace trix {

template <typename V>
concept VectorConcept = requires(V const v, size_t i) {
  typename V::value_type;
  { v.operator[](i) } -> std::same_as<typename V::value_type>;
  { V::components } -> std::convertible_to<size_t const>;
  { V::elements } -> std::convertible_to<size_t const>;
};

template <typename V>
concept MutableVectorConcept = VectorConcept<V> && requires(V v, size_t i) {
  { v.operator[](i) } -> std::same_as<typename V::value_type&>;
};

template <size_t N, typename T>
struct VectorStorageType {
  static constexpr size_t const components = N;
  static constexpr size_t elements = 0;
  using value_type = T;
};

template <size_t N, typename T>
struct FullVectorType {
  template <typename F, size_t... Is>
    requires std::same_as<std::invoke_result_t<F, size_t>, void>
  constexpr void for_each_element(F fun, std::index_sequence<Is...>) {
    (fun(Is), ...);
  }

  template <typename F>
    requires std::same_as<std::invoke_result_t<F, size_t>, void>
  constexpr void for_each_element(F fun) {
    for_each_element(fun, std::make_index_sequence<N>{});
  }

  template <typename F, size_t... Is>
    requires std::same_as<std::invoke_result_t<F, size_t>, T>
  constexpr auto sum_for_each_element(F fun, std::index_sequence<Is...>) const {
    return (... + fun(Is));
  }

  template <typename F>
    requires std::same_as<std::invoke_result_t<F, size_t>, T>
  constexpr auto sum_for_each_element(F fun) const {
    return sum_for_each_element(fun, std::make_index_sequence<N>{});
  }
};

template <typename STORAGE, size_t N, size_t SIZE, typename T = Number>
struct VectorArrayStorage : VectorStorageType<N, T>, FullVectorType<N, T> {
  static constexpr size_t elements = SIZE;
  template <std::convertible_to<T>... Values>
  explicit constexpr VectorArrayStorage(Values&&... values)
      : a_{std::forward<Values>(values)...} {};
  explicit constexpr VectorArrayStorage(std::array<T, elements>&& array)
      : a_{std::move(array)} {};
  explicit constexpr VectorArrayStorage(std::array<T, elements> const& array)
      : a_{array} {};
  template <std::input_iterator It, typename Ite>
    requires std::convertible_to<typename std::iterator_traits<It>::value_type,
                                 T>
  constexpr VectorArrayStorage(It first, Ite last) {
    for (size_t i = 0; i < SIZE; ++i) {
      if (first == last)
        break;
      a_[i] = *first++;
    }
  }
  template <std::ranges::input_range R>
  constexpr VectorArrayStorage(std::from_range_t, R const& r)
      : VectorArrayStorage(r.begin(), r.end()) {}
  constexpr T operator[](size_t const i) const {
    return a_[check_and_get_offset_(i)];
  }
  constexpr T& operator[](size_t const i) {
    return a_[check_and_get_offset_(i)];
  }
  constexpr size_t size() const {
    return elements;
  }

private:
  std::array<T, elements> a_{};
  static constexpr size_t check_and_get_offset_(size_t const i) {
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
  static constexpr size_t get_offset(size_t const i) {
    return i;
  }
};

template <size_t N, typename T = Number,
          template <size_t, typename C> typename Storage = VectorGenericStorage>
  requires(N > 0)
struct Vector;

template <VectorConcept V, size_t B, size_t E = V::components, size_t S = 1>
struct Slice : VectorStorageType<(E - B - 1) / S + 1, typename V::value_type> {
  static constexpr size_t const start = B;
  static constexpr size_t const stop = E;
  static constexpr size_t const stride = S;

  constexpr Slice(V& vector) : vector_(vector) {}

  constexpr auto operator[](size_t const i) const {
    size_t const offset = start + i * stride;
    assert(offset < stop);
    return vector_[offset];
  }

  constexpr auto& operator[](size_t const i) {
    size_t const offset = start + i * stride;
    assert(offset < stop);
    return vector_[offset];
  }

private:
  V& vector_;
};

template <VectorConcept V1, VectorConcept V2>
struct CommonVector {
  using type =
      Vector<V1::components, std::common_type_t<typename V1::value_type,
                                                typename V2::value_type>>;
};

template <VectorConcept V1, VectorConcept V2>
using common_vector_t = CommonVector<V1, V2>::type;

template <VectorConcept V, typename R = V>
struct VectorUnaryOperation
    : VectorStorageType<V::components, typename V::value_type> {
  using arg_type = V;
  using result_type = R;
  explicit constexpr VectorUnaryOperation(V const& arg) : arg{arg} {}
  constexpr operator result_type(this auto&& self) {
    return result_type{self};
  }

protected:
  V const& arg;
};

template <VectorConcept V>
struct VectorNegation : VectorUnaryOperation<V> {
  using VectorUnaryOperation<V>::VectorUnaryOperation;
  constexpr auto operator[](size_t const i) const {
    return -(this->arg[i]);
  }
};

template <VectorConcept V1, VectorConcept V2,
          typename R = common_vector_t<V1, V2>>
struct VectorBinaryOperation
    : VectorStorageType<V1::components,
                        std::common_type_t<typename V1::value_type,
                                           typename V2::value_type>> {
  using arg1_type = V1;
  using arg2_type = V2;
  using result_type = R;
  explicit constexpr VectorBinaryOperation(V1 const& arg1, V2 const& arg2)
      : arg1{arg1}, arg2{arg2} {}
  constexpr operator result_type(this auto&& self) {
    return result_type{self};
  }

protected:
  V1 const& arg1;
  V2 const& arg2;
};

template <VectorConcept V1, VectorConcept V2>
struct VectorAddition : VectorBinaryOperation<V1, V2> {
  using VectorBinaryOperation<V1, V2>::VectorBinaryOperation;
  constexpr auto operator[](size_t const i) const {
    return this->arg1[i] + this->arg2[i];
  }
};

template <VectorConcept V1, VectorConcept V2>
struct VectorSubtraction : VectorBinaryOperation<V1, V2> {
  using VectorBinaryOperation<V1, V2>::VectorBinaryOperation;
  constexpr auto operator[](size_t const i) const {
    return this->arg1[i] - this->arg2[i];
  }
};

template <size_t N, typename T, template <size_t, typename C> typename Storage>
  requires(N > 0)
struct Vector : Storage<N, T> {
  using Storage<N, T>::Storage;
  template <VectorConcept O>
    requires(O::components >= Vector::components)
  explicit constexpr Vector(O const& other)
      : Storage<O::components, typename O::value_type>{} {
    this->for_each_element([this, &other](size_t i) { (*this)[i] = other[i]; });
  }

  template <VectorConcept O>
    requires(O::components == N)
  constexpr Vector& operator+=(O const& other) {
    this->for_each_element(
        [this, &other](size_t i) { (*this)[i] += other[i]; });
    return *this;
  }

  template <VectorConcept O>
    requires(O::components == N)
  constexpr Vector& operator-=(O const& other) {
    this->for_each_element(
        [this, &other](size_t i) { (*this)[i] -= other[i]; });
    return *this;
  }

  constexpr Vector& operator*=(ScalarConcept auto const value) {
    this->for_each_element([this, value](size_t i) { (*this)[i] *= value; });
    return *this;
  }

  constexpr auto operator-() const {
    return VectorNegation<Vector>{*this};
  }

  constexpr T norm() const {
    return std::sqrt(this->sum_for_each_element(
        [this](size_t i) { return (*this)[i] * (*this)[i]; }));
  }

  constexpr auto length() const {
    return norm();
  }

  template <size_t B, size_t E = N, size_t S = 1>
  constexpr auto slice(this auto&& self) {
    return Slice<std::remove_reference_t<decltype(self)>, B, std::min(E, N), S>{
        self};
  }
};

template <VectorConcept V>
constexpr auto begin(V const& v) {
  return IndexIterator(v, 0);
}

template <VectorConcept V>
constexpr auto end(V const& v) {
  return IndexIterator(v, V::components);
}

template <typename C, typename... Cs>
auto vector(C&& first, Cs&&... components) {
  return Vector<sizeof...(Cs) + 1, C>{std::forward<C>(first),
                                      std::forward<Cs>(components)...};
}

template <VectorConcept V1, VectorConcept V2, size_t... Is>
constexpr bool equals(V1 const& v1, V2 const& v2, std::index_sequence<Is...>) {
  return (... && (v1[Is] == v2[Is]));
}

template <VectorConcept V1, VectorConcept V2>
  requires(V1::components == V2::components)
constexpr bool operator==(V1 const& v1, V2 const& v2) {
  return equals(v1, v2, std::make_index_sequence<V1::components>{});
}

template <VectorConcept V1, VectorConcept V2>
  requires(V1::components == V2::components)
constexpr auto operator+(V1 const& v1, V2 const& v2) {
  Vector<V1::components,
         std::common_type_t<typename V1::value_type, typename V2::value_type>>
      result{v1};
  result += v2;
  return result;
}

template <VectorConcept V1, VectorConcept V2>
  requires(V1::components == V2::components)
constexpr auto operator-(V1 const& v1, V2 const& v2) {
  Vector<V1::components,
         std::common_type_t<typename V1::value_type, typename V2::value_type>>
      result{v1};
  result -= v2;
  return result;
}

template <size_t N, typename T, template <size_t, typename> typename S,
          ScalarConcept V>
  requires MutableVectorConcept<Vector<N, T, S>>
constexpr auto operator*(Vector<N, T, S> const& v, V const value) {
  Vector<N, T, S> result{v};
  result *= value;
  return result;
}

template <VectorConcept V, ScalarConcept S>
constexpr auto operator*(V const& v, S const value) {
  Vector<V::components, typename V::value_type> result{v};
  result *= value;
  return result;
}

template <VectorConcept V, ScalarConcept S>
constexpr auto operator*(S const value, V const& v) {
  return v * value;
}

template <VectorConcept V1, VectorConcept V2, size_t... Is>
  requires(V1::components == V2::components)
constexpr auto dot(V1 const& v1, V2 const& v2, std::index_sequence<Is...>) {
  std::common_type_t<typename V1::value_type, typename V2::value_type> result{
      (... + (v1[Is] * v2[Is]))};
  return result;
}

template <VectorConcept V1, VectorConcept V2>
  requires(V1::components == V2::components)
constexpr auto operator*(V1 const& v1, V2 const& v2) {
  return dot(v1, v2, std::make_index_sequence<V1::components>{});
}

template <VectorConcept V1, VectorConcept V2>
  requires(V1::components == 3 && V2::components == 3)
constexpr auto cross(V1 const& v1, V2 const& v2) {
  return vector(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0]);
}

} // namespace trix

#endif
