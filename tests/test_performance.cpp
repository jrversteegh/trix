#include <iostream>
#include <random>
#include <cassert>

#include <benchmark/benchmark.h>

#include "trix/vector.h"
#include "trix/matrix.h"

using namespace trix;

using trix::operator==;

auto r() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<Number> dis(0.0, 1.0);
  return dis(gen);
}

auto get_random_matrix() {
  return matrix(
    r(), r(), r(),
    r(), r(), r(),
    r(), r(), r()
  );
}

auto get_random_symmetric() {
  return SymmetricMatrix<3>{
    r(),
    r(), r(),
    r(), r(), r(),
  };
}

auto get_random_diagonal() {
  return DiagonalMatrix<3>{
    r(), r(), r(),
  };
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
    // Number of loops tuned so it takes ~100ns on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 29; ++i) {
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
    // Number of loops tuned so it takes ~100ns on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 35; ++i) {
      auto value = m1 == m2;
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
    // Number of loops tuned so it takes ~100ns on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 51; ++i) {
      auto value = m1 == m2;
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
    // Number of loops tuned so it takes ~100ns on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 99; ++i) {
      auto value = m1 == m2;
      benchmark::DoNotOptimize(value);
    }
  }
}

BENCHMARK(benchmark_matrix_mul);
BENCHMARK(benchmark_matrix_equality);
BENCHMARK(benchmark_symmetric_equality);
BENCHMARK(benchmark_diagonal_equality);

static void benchmark_vector_op_minus(benchmark::State& state) {
  auto v = get_random_vector();
  benchmark::DoNotOptimize(v);
  for (auto _ : state) {
    // Number of loops tuned so it takes ~100ns on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 130; ++i) {
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
    // Number of loops tuned so it takes ~100ns on i9 13k9 compiled with GCC 15.1
    for (int i = 0; i < 30; ++i) {
      auto value = v1 == v2;
      benchmark::DoNotOptimize(value);
    }
  }
}


BENCHMARK(benchmark_vector_op_minus);
BENCHMARK(benchmark_vector_equality);

BENCHMARK_MAIN();
