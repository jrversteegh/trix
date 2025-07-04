#ifndef TRIX_UTILS_H__
#define TRIX_UTILS_H__

#include "config.h"

namespace trix {

template <size_t S>
constexpr size_t size_t_sqrt() {
  size_t result = 0;
  while (result * result < S)
    ++result;
  return result;
}

} // namespace trix

#endif
