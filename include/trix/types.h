#ifndef TRIX_TYPES_H__
#define TRIX_TYPES_H__

#include "config.h"

namespace trix {

template <typename T>
struct IndexIterator {
  using difference_type = typename T::value_type;
  using value_type = typename T::value_type;
  using pointer = value_type const*;
  using reference = value_type;
  using iterator_category = std::input_iterator_tag;

  constexpr IndexIterator(T const& container, size_t pos)
      : container_(container), pos_(pos) {}

  constexpr reference operator*() const {
    return container_[pos_];
  }

  constexpr IndexIterator& operator++() {
    ++pos_;
    return *this;
  }
  constexpr IndexIterator operator++(int) {
    auto result = *this;
    ++(*this);
    return result;
  }

  constexpr bool operator==(IndexIterator const& other) const {
    return pos_ == other.pos_ && &container_ == &other.container_;
  }

protected:
  T const& container_;
  size_t pos_;
};

template <typename T, size_t S>
struct DoubleIndexIterator : IndexIterator<T> {
  constexpr IndexIterator<T>::value_type operator*() const {
    return this->container_[this->pos_ / S, this->pos_ % S];
  }
};

} // namespace trix

#endif
