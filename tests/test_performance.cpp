#include <cassert>
#include <iostream>
#include <random>

#include <benchmark/benchmark.h>

#include "trix/config.h"
#include "trix/matrix.h"
#include "trix/printing.h"
#include "trix/types.h"
#include "trix/vector.h"

#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#endif

using namespace trix;

auto r() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<Number> dis(0.0, 1.0);
  return dis(gen);
}

struct RandomGenerator {
  using value_type = Number;
  Number operator[](size_t) const {
    return r();
  }
};

template <size_t N = 4>
auto get_random_matrix() {
  auto gen = RandomGenerator();
  return Matrix<N, N>(IndexIterator{gen, 0}, IndexIterator{gen, N * N});
}

#ifdef HAVE_EIGEN
template <size_t N = 4>
auto get_random_eigen() {
  Eigen::Matrix<Number, N, N, Eigen::RowMajor> result;
  result << r(), r(), r(), r(), r(), r(), r(), r(), r(), r(), r(), r(), r(),
      r(), r(), r();
  return result;
}
#endif

template <size_t N = 4>
auto get_random_symmetric() {
  auto gen = RandomGenerator();
  return SymmetricMatrix<N>{IndexIterator{gen, 0}, IndexIterator{gen, N * N}};
}

template <size_t N = 4>
auto get_random_diagonal() {
  auto gen = RandomGenerator();
  return DiagonalMatrix<N>{IndexIterator{gen, 0}, IndexIterator{gen, N}};
}

template <size_t N = 10>
auto get_random_vector() {
  auto gen = RandomGenerator();
  return Vector<N>(IndexIterator{gen, 0}, IndexIterator{gen, N});
}

static void benchmark_matrix_mul(benchmark::State& state) {
  auto m1 = get_random_matrix();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_matrix();
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    decltype(m1) value = m1 * m2;
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(benchmark_matrix_mul);

#ifdef HAVE_EIGEN
static void benchmark_eigen_mul(benchmark::State& state) {
  auto m1 = get_random_eigen();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_eigen();
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    decltype(m1) value = m1 * m2;
    benchmark::DoNotOptimize(value);
  }
  decltype(m1) value = m1 * m2;
  auto t1 = Matrix<m1.rows(), m1.cols()>{m1.data(), m1.data() + m1.size()};
  auto t2 = Matrix<m2.rows(), m2.cols()>{m2.data(), m2.data() + m2.size()};
  auto result = Matrix<value.rows(), value.cols()>{value.data(),
                                                   value.data() + value.size()};
  auto expected = t1 * t2;
  assert(all_close(result, expected));
}
BENCHMARK(benchmark_eigen_mul);
#endif

static void benchmark_matrix_equality(benchmark::State& state) {
  auto m1 = get_random_matrix();
  benchmark::DoNotOptimize(m1);
  auto m2 = m1;
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    auto value = m1 == m2;
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(benchmark_matrix_equality);

static void benchmark_symmetric_mul(benchmark::State& state) {
  auto m1 = get_random_symmetric();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_symmetric();
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    auto value = (m1 * m2)();
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(benchmark_symmetric_mul);

static void benchmark_symmetric_equality(benchmark::State& state) {
  auto m1 = get_random_symmetric();
  benchmark::DoNotOptimize(m1);
  auto m2 = m1;
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    auto value = m1 == m2;
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(benchmark_symmetric_equality);

static void benchmark_diagonal_mul(benchmark::State& state) {
  auto m1 = get_random_diagonal();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_diagonal();
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    auto value = (m1 * m2)();
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(benchmark_diagonal_mul);

static void benchmark_diagonal_equality(benchmark::State& state) {
  auto m1 = get_random_diagonal();
  benchmark::DoNotOptimize(m1);
  auto m2 = m1;
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    auto value = m1 == m2;
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(benchmark_diagonal_equality);

static void benchmark_vector_op_minus(benchmark::State& state) {
  auto v = get_random_vector();
  benchmark::DoNotOptimize(v);
  for (auto _ : state) {
    auto value = (-v)();
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(benchmark_vector_op_minus);

static void benchmark_vector_op_cross(benchmark::State& state) {
  auto v1 = get_random_vector<3>();
  benchmark::DoNotOptimize(v1);
  auto v2 = get_random_vector<3>();
  benchmark::DoNotOptimize(v2);
  for (auto _ : state) {
    auto value = cross(v1, v2)();
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(benchmark_vector_op_cross);

static void benchmark_vector_equality(benchmark::State& state) {
  auto v1 = get_random_vector();
  benchmark::DoNotOptimize(v1);
  auto v2 = v1;
  benchmark::DoNotOptimize(v2);
  for (auto _ : state) {
    bool value = v1 == v2;
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(benchmark_vector_equality);

constexpr size_t size = 128;

static void benchmark_matrix_star_operator(benchmark::State& state) {
  auto m1 = get_random_matrix<size>();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_matrix<size>();
  assert(!all_close(m1, m2, 1E-10));
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    auto value = (m1 * m2)();
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(benchmark_matrix_star_operator);

static void benchmark_matrix_strassen(benchmark::State& state) {
  auto m1 = get_random_matrix<size>();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_matrix<size>();
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    auto value = strassen(m1, m2);
    benchmark::DoNotOptimize(value);
  }
  auto m = m1 * m2;
  auto s = strassen(m1, m2);
  assert(all_close(m, s, 1E-10));
}
BENCHMARK(benchmark_matrix_strassen);

static void benchmark_matrix_block_mul(benchmark::State& state) {
  auto m1 = get_random_matrix<size>();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_matrix<size>();
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    auto value = block_mul(m1, m2);
    benchmark::DoNotOptimize(value);
  }
  auto m = m1 * m2;
  auto s = block_mul(m1, m2);
  assert(all_close(m, s, 1E-10));
}
BENCHMARK(benchmark_matrix_block_mul);

static void benchmark_matrix_blas_mul(benchmark::State& state) {
  auto m1 = get_random_matrix<size>();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_matrix<size>();
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    auto value = blas_mul(m1, m2);
    benchmark::DoNotOptimize(value);
  }
  auto m = m1 * m2;
  auto s = blas_mul(m1, m2);
  assert(all_close(m, s, 1E-10));
}
BENCHMARK(benchmark_matrix_blas_mul);

BENCHMARK_MAIN();
