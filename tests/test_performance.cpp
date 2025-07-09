#include <cassert>
#include <iostream>
#include <random>

#include <benchmark/benchmark.h>

#include "trix/matrix.h"
#include "trix/types.h"
#include "trix/vector.h"

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

auto get_random_vector() {
  return vector(r(), r(), r(), r(), r(), r(), r(), r(), r(), r());
}

static void benchmark_matrix_mul(benchmark::State& state) {
  auto m1 = get_random_matrix();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_matrix();
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    // Number of loops tuned so it takes ~1mus on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 190; ++i) {
      auto value = m1 * m2;
      benchmark::DoNotOptimize(value);
    }
  }
}

static void benchmark_matrix_equality(benchmark::State& state) {
  auto m1 = get_random_matrix();
  benchmark::DoNotOptimize(m1);
  auto m2 = m1;
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    // Number of loops tuned so it takes ~1mus on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 206; ++i) {
      auto value = m1 == m2;
      benchmark::DoNotOptimize(value);
    }
  }
}

static void benchmark_symmetric_mul(benchmark::State& state) {
  auto m1 = get_random_symmetric();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_symmetric();
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    // Number of loops tuned so it takes ~1mus on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 170; ++i) {
      auto value = m1 * m2;
      benchmark::DoNotOptimize(value);
    }
  }
}

static void benchmark_symmetric_equality(benchmark::State& state) {
  auto m1 = get_random_symmetric();
  benchmark::DoNotOptimize(m1);
  auto m2 = m1;
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    // Number of loops tuned so it takes ~1mus on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 325; ++i) {
      auto value = m1 == m2;
      benchmark::DoNotOptimize(value);
    }
  }
}

static void benchmark_diagonal_mul(benchmark::State& state) {
  auto m1 = get_random_diagonal();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_diagonal();
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    // Number of loops tuned so it takes ~1mus on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 2765; ++i) {
      auto value = m1 * m2;
      benchmark::DoNotOptimize(value);
    }
  }
}

static void benchmark_diagonal_equality(benchmark::State& state) {
  auto m1 = get_random_diagonal();
  benchmark::DoNotOptimize(m1);
  auto m2 = m1;
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    // Number of loops tuned so it takes ~1mus on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 780; ++i) {
      auto value = m1 == m2;
      benchmark::DoNotOptimize(value);
    }
  }
}

BENCHMARK(benchmark_matrix_mul);
BENCHMARK(benchmark_matrix_equality);
BENCHMARK(benchmark_symmetric_mul);
BENCHMARK(benchmark_symmetric_equality);
BENCHMARK(benchmark_diagonal_mul);
BENCHMARK(benchmark_diagonal_equality);

static void benchmark_vector_op_minus(benchmark::State& state) {
  auto v = get_random_vector();
  benchmark::DoNotOptimize(v);
  for (auto _ : state) {
    // Number of loops tuned so it takes ~1mus on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 1300; ++i) {
      auto value = -v;
      benchmark::DoNotOptimize(value);
    }
  }
}

static void benchmark_vector_equality(benchmark::State& state) {
  auto v1 = get_random_vector();
  benchmark::DoNotOptimize(v1);
  auto v2 = v1;
  benchmark::DoNotOptimize(v2);
  for (auto _ : state) {
    // Number of loops tuned so it takes ~1mus on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 322; ++i) {
      auto value = v1 == v2;
      benchmark::DoNotOptimize(value);
    }
  }
}

BENCHMARK(benchmark_vector_op_minus);
BENCHMARK(benchmark_vector_equality);

constexpr size_t size = 128;

static void benchmark_matrix_star_operator(benchmark::State& state) {
  auto m1 = get_random_matrix<size>();
  benchmark::DoNotOptimize(m1);
  auto m2 = get_random_matrix<size>();
  assert(!all_close(m1, m2, 1E-10));
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    auto value = m1 * m2;
    benchmark::DoNotOptimize(value);
  }
}

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
  assert(m != s);
  assert(all_close(m, s, 1E-10));
}

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
  assert(m != s);
  assert(all_close(m, s, 1E-10));
}

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
  assert(m != s);
  assert(all_close(m, s, 1E-10));
}

BENCHMARK(benchmark_matrix_star_operator);
BENCHMARK(benchmark_matrix_strassen);
BENCHMARK(benchmark_matrix_block_mul);
BENCHMARK(benchmark_matrix_blas_mul);

BENCHMARK_MAIN();
