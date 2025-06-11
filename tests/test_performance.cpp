#include <iostream>
#include <random>

#include <benchmark/benchmark.h>

#include "trix/matrix.h"

using namespace trix;

auto r() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<Number> dis(0.0, 1.0);
  return dis(gen);
}

auto get_random_matrix() {
  return Matrix<3, 3>{ 
    r(), r(), r(),
    r(), r(), r(),
    r(), r(), r(),
  };
}

static void benchmark_matrix_mul(benchmark::State& state) {
  auto m1 = get_random_matrix();
  auto m2 = get_random_matrix();
  benchmark::DoNotOptimize(m1);
  benchmark::DoNotOptimize(m2);
  for (auto _ : state) {
    auto value = m1 * m2;
    benchmark::DoNotOptimize(value);
  }
}

BENCHMARK(benchmark_matrix_mul);
BENCHMARK_MAIN();
