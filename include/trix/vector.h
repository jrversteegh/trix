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
concept VectorConcept = requires(noref<V> const v, size_t i) {
  typename noref<V>::value_type;
  typename noref<V>::result_type;
  { v.operator[](i) } -> std::same_as<typename noref<V>::value_type>;
  { noref<V>::components } -> std::convertible_to<size_t const>;
  { noref<V>::elements } -> std::convertible_to<size_t const>;
};

template <typename V>
concept MutableVectorConcept =
    VectorConcept<V> && requires(noref<V> v, size_t i) {
      { v.operator[](i) } -> std::same_as<typename noref<V>::value_type&>;
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
  template <typename It, typename Ite>
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

template <VectorConcept V, size_t B, size_t E = V::components, size_t S = 1,
          size_t C = (E - B - 1) / S + 1>
struct Slice : VectorStorageType<C, typename V::value_type> {
  static constexpr size_t const start = B;
  static constexpr size_t const stop = E;
  static constexpr size_t const stride = S;
  using result_type = V::template resized<C>;

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
using common_vector_t = CommonVector<noref<V1>, noref<V2>>::type;

template <VectorConcept V, typename R = noref<V>>
struct VectorUnaryOperation
    : VectorStorageType<noref<V>::components, typename noref<V>::value_type> {
  using arg_type = make_const<V>;
  using result_type = R;
  VectorUnaryOperation() = delete;
  explicit constexpr VectorUnaryOperation(V&& arg)
      : arg{std::forward<V>(arg)} {}
  constexpr operator result_type(this auto&& self) {
    return result_type{self};
  }
  constexpr auto operator()(this auto&& self) {
    return static_cast<result_type>(self);
  }
  arg_type arg;
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
    : VectorStorageType<noref<V1>::components,
                        std::common_type_t<typename noref<V1>::value_type,
                                           typename noref<V2>::value_type>> {
  using arg1_type = make_const<V1>;
  using arg2_type = make_const<V2>;
  using result_type = R;
  constexpr VectorBinaryOperation(V1&& arg1, V2&& arg2)
      : arg1{std::forward<V1>(arg1)}, arg2{std::forward<V2>(arg2)} {}
  constexpr operator result_type(this auto&& self) {
    return result_type{self};
  }
  constexpr auto operator()(this auto&& self) {
    return static_cast<result_type>(self);
  }

protected:
  arg1_type arg1;
  arg2_type arg2;
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

template <VectorConcept V1, VectorConcept V2>
  requires(noref<V1>::components == 3 && noref<V1>::components == 3)
struct VectorCross : VectorBinaryOperation<V1, V2> {
  using VectorBinaryOperation<V1, V2>::VectorBinaryOperation;
  constexpr auto operator[](size_t const i) const {
    size_t j = (i + 1) % 3;
    size_t k = (i + 2) % 3;
    return this->arg1[j] * this->arg2[k] - this->arg1[k] * this->arg2[j];
  }
};

template <VectorConcept V, ScalarConcept S,
          typename R = noref<V>::template retyped<
              std::common_type_t<typename noref<V>::value_type, S>>>
struct VectorMultiplication
    : VectorStorageType<noref<V>::components, typename noref<V>::value_type> {
  using arg1_type = make_const<V>;
  using arg2_type = make_const<S>;
  using result_type = R;
  explicit constexpr VectorMultiplication(V&& arg1, S const arg2)
      : arg1{std::forward<V>(arg1)}, arg2{arg2} {}
  constexpr operator result_type() {
    return result_type{*this};
  }
  constexpr auto operator[](size_t const i) const {
    return arg1[i] * arg2;
  }
  constexpr auto operator()() {
    return static_cast<result_type>(*this);
  }

private:
  arg1_type arg1;
  arg2_type arg2;
};

template <size_t N, typename T, template <size_t, typename C> typename Storage>
  requires(N > 0)
struct Vector : Storage<N, T> {
  using storage = Storage<N, T>;
  using storage::storage;
  using result_type = Vector;
  template <size_t P>
  using resized = Vector<P, T, Storage>;
  template <typename V>
  using retyped = Vector<N, V, Storage>;
  template <VectorConcept O>
    requires(O::components >= Vector::elements)
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

  constexpr T norm() const {
    return std::sqrt(this->sum_for_each_element(
        [this](size_t i) { return (*this)[i] * (*this)[i]; }));
  }

  constexpr auto length() const {
    return norm();
  }

  template <size_t B, size_t E = N, size_t S = 1>
  constexpr auto slice(this auto&& self) {
    return Slice<noref<decltype(self)>, B, std::min(E, N), S>{self};
  }
};

template <VectorConcept V>
constexpr auto operator-(V&& v) {
  return VectorNegation<V>{std::forward<V>(v)};
}

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
  requires(noref<V1>::components == noref<V2>::components)
constexpr auto operator+(V1&& v1, V2&& v2) {
  return VectorAddition<V1, V2>{std::forward<V1>(v1), std::forward<V2>(v2)};
}

template <VectorConcept V1, VectorConcept V2>
  requires(noref<V1>::components == noref<V2>::components)
constexpr auto operator-(V1&& v1, V2&& v2) {
  return VectorSubtraction<V1, V2>{std::forward<V1>(v1), std::forward<V2>(v2)};
}

template <VectorConcept V, ScalarConcept S>
constexpr auto operator*(V&& v, S const value) {
  return VectorMultiplication<V, S>{std::forward<V>(v), value};
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
  requires(noref<V1>::components == 3 && noref<V2>::components == 3)
constexpr auto cross(V1&& v1, V2&& v2) {
  return VectorCross<V1, V2>{std::forward<V1>(v1), std::forward<V2>(v2)};
}

} // namespace trix

#endif
