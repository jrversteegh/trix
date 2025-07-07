#ifndef TRIX_PRINTING_H__
#define TRIX_PRINTING_H__

#include <array>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "config.h"
#include "matrix.h"
#include "vector.h"

namespace trix {

template <VectorConcept C>
constexpr std::string to_string(C const& c) {
  std::vector<typename C::value_type> values{};
  for (size_t i = 0; i < C::components; ++i) {
    values.push_back(c[i]);
  }
  return fmt::format(trix_fmtstr, fmt::join(values, ", "));
}

template <MatrixConcept M>
constexpr std::string to_string(M const& m) {
  std::array<std::string, M::rows> lines{};
  for (size_t i = 0; i < M::rows; ++i) {
    std::vector<typename M::value_type> values{};
    for (size_t j = 0; j < M::columns; ++j) {
      values.push_back(m[i, j]);
    }
    lines[i] = fmt::format(trix_fmtstr, fmt::join(values, ", "));
  }
  return fmt::format("{}", fmt::join(lines, "\n"));
}

template <typename T>
concept HasToString = requires(T v) { to_string(v); };

template <typename C>
  requires HasToString<C>
std::ostream& operator<<(std::ostream& out, C const& c) {
  out << to_string(c);
  return out;
}

} // namespace trix

#endif
