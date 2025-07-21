#include "test_common.h"

#include <ranges>

#include <boost/test/parameterized_test.hpp>

#include "trix/matrix.h"
#include "trix/printing.h"

namespace ut = boost::unit_test;
namespace ranges = std::ranges;

using namespace trix;

BOOST_AUTO_TEST_CASE(matrix_construct_from_iter_test) {
  auto source = std::vector{1., 2., 3., 4.};
  auto m1 = Matrix<2, 2>(source.begin(), source.end());
  BOOST_TEST(m1 == matrix(1., 2., 3., 4.));

  double psource[] = {1., 2., 3., 4.};
  auto m2 = Matrix<2, 2>(psource, psource + 4);
  BOOST_TEST(m2 == matrix(1., 2., 3., 4.));
}

BOOST_AUTO_TEST_CASE(matrix_construct_from_array_test) {
  auto m1 = Matrix<2, 2>(std::array{1., 2., 3., 4.});
  BOOST_TEST(m1 == matrix(1., 2., 3., 4.));

  auto source = std::array{1., 2., 3., 4.};
  auto m2 = Matrix<2, 2>(source);
  BOOST_TEST(m2 == matrix(1., 2., 3., 4.));
}

BOOST_AUTO_TEST_CASE(matrix_construct_from_range_test) {
  auto const a = std::array{1., 2., 3., 4.};
  auto m1 = Matrix<2, 2>(std::from_range, a);
  BOOST_TEST(m1 == matrix(1., 2., 3., 4.));

  m1 = Matrix<2, 2>(std::from_range, a | ranges::views::take(3));
  BOOST_TEST(m1 == matrix(1., 2., 3., 0.));

  auto const to_short = std::array{1., 2.};
  auto m2 = Matrix<2, 2>(std::from_range, to_short);
  BOOST_TEST(m2 == matrix(1., 2., 0., 0.));

  auto m3 = Matrix<2, 2>(std::from_range, ranges::views::iota(1));
  BOOST_TEST(m3 == matrix(1., 2., 3., 4.));
}

struct MatrixFixture {
  Matrix<3, 4> m34{
      1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0,
  };
  Matrix<4, 3> m43{
      1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0,
  };
  Matrix<3, 3> e33{
      30.0, 40.0, 50.0, 40.0, 54.0, 68.0, 50.0, 68.0, 86.0,
  };
  Matrix<4, 4> e44{
      14.0, 20.0, 26.0, 32.0, 20.0, 29.0, 38.0, 47.0,
      26.0, 38.0, 50.0, 62.0, 32.0, 47.0, 62.0, 77.0,
  };
  Vector<3> v3{1., 2., 3.};
  Vector<4> v4{1., 2., 3., 4.};
};

BOOST_FIXTURE_TEST_CASE(matrix_mul_test3, MatrixFixture) {
  auto r = m34 * m43;
  BOOST_TEST(r.rows == 3);
  BOOST_TEST(r.columns == 3);
  BOOST_TEST(r == e33);
};

BOOST_FIXTURE_TEST_CASE(matrix_mul_test4, MatrixFixture) {
  auto r = m43 * m34;
  BOOST_TEST(r.rows == 4);
  BOOST_TEST(r.columns == 4);
  BOOST_TEST(r == e44);
};

BOOST_FIXTURE_TEST_CASE(matrix_vector_pre_mul, MatrixFixture) {
  auto r = v4 * m43;
  BOOST_TEST(r.components == 3);
  BOOST_TEST(r == vector(30., 40., 50.));
};

BOOST_FIXTURE_TEST_CASE(matrix_vector_post_mul, MatrixFixture) {
  auto r = m34 * v4;
  BOOST_TEST(r.components == 3);
  BOOST_TEST(r == vector(30., 40., 50.));
};

BOOST_FIXTURE_TEST_CASE(matrix_scalar_mul_test, MatrixFixture) {
  auto r1 = 2 * m34;
  auto r2 = m34 * 2;
  auto e = m34 + m34;
  BOOST_TEST(r1 == e);
  BOOST_TEST(r2 == e);
};

struct SymmetricMatrixFixture {

  SymmetricMatrix<3> symmetric{
      1.0, 2.0, 3.0, 3.0, 4.0, 5.0,
  };

  Matrix<3, 3> square{symmetric};
  Matrix<3, 3> square_asymmetric{
      1.0, 2.0, 4.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0,
  };

  SymmetricMatrix<3> symmetric_multiplied{
      14., 20., 29., 26., 38., 50.,
  };

  Matrix<3, 3> square_multiplied{symmetric_multiplied};

  SymmetricMatrix<3> symmetric_added{2 * symmetric};
  Matrix<3, 3> square_added{symmetric_added};

  SymmetricMatrix<3> symmetric_zero{};
  Matrix<3, 3> square_zero{};
};

BOOST_FIXTURE_TEST_CASE(matrix_construction_helper_test,
                        SymmetricMatrixFixture){
    // auto r = matrix(1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0);
    // BOOST_TEST(r == symmetric);
};

template <typename ARG1, typename ARG2>
struct Add {
  auto operator()(ARG1 const& arg1, ARG2 const& arg2) {
    return arg1 + arg2;
  }
};

template <typename ARG1, typename ARG2>
struct Subtract {
  auto operator()(ARG1 const& arg1, ARG2 const& arg2) {
    return arg1 - arg2;
  }
};

template <typename ARG1, typename ARG2>
struct Multiply {
  auto operator()(ARG1 const& arg1, ARG2 const& arg2) {
    return arg1 * arg2;
  }
};

template <template <typename, typename> typename OPERATION>
struct OperationTest {
  template <typename ARG1, typename ARG2, typename EXPECTED>
  void operator()(ARG1 const& arg1, ARG2 const& arg2,
                  EXPECTED const& expected) {
    OPERATION<ARG1, ARG2> operation{};
    auto r = operation(arg1, arg2);
    BOOST_TEST((typeid(decltype(r())) == typeid(expected)));
    BOOST_TEST(r == expected);
  }
};

void do_cross_test(auto& test, auto arg1, auto arg2, auto expected1,
                   auto expected2) {
  test(arg1, arg1, expected1);
  test(arg1, arg2, expected1);
  test(arg2, arg1, expected1);
  test(arg2, arg2, expected2);
}

BOOST_FIXTURE_TEST_CASE(matrix_symmetric_equality, SymmetricMatrixFixture) {
  BOOST_TEST(symmetric == symmetric);
  BOOST_TEST(square == symmetric);
  BOOST_TEST(symmetric == square);
  BOOST_TEST(symmetric != square_asymmetric);
};

BOOST_FIXTURE_TEST_CASE(matrix_symmetric_multiply, SymmetricMatrixFixture) {
  OperationTest<Multiply> test{};
  do_cross_test(test, square, symmetric, square_multiplied,
                symmetric_multiplied);
};

BOOST_FIXTURE_TEST_CASE(matrix_symmetric_add, SymmetricMatrixFixture) {
  OperationTest<Add> test{};
  do_cross_test(test, square, symmetric, square_added, symmetric_added);
};

BOOST_FIXTURE_TEST_CASE(matrix_symmetric_subtract, SymmetricMatrixFixture) {
  OperationTest<Subtract> test{};
  do_cross_test(test, square, symmetric, square_zero, symmetric_zero);
};

BOOST_FIXTURE_TEST_CASE(matrix_transpose, MatrixFixture) {
  BOOST_TEST(m43.transpose() == m34);
  BOOST_TEST((m43 * m43.transpose()) == (m43 * m34));
  BOOST_TEST((m34 + m43.transpose()) == (2 * m34));
}

BOOST_FIXTURE_TEST_CASE(matrix_diagonal, MatrixFixture) {
  BOOST_TEST(m43.diagonal() == vector(1., 3., 5.));
}

BOOST_FIXTURE_TEST_CASE(matrix_row, MatrixFixture) {
  BOOST_TEST(m43.row(0) == vector(1., 2., 3.));
}

BOOST_FIXTURE_TEST_CASE(matrix_column, MatrixFixture) {
  BOOST_TEST(m43.column(0) == vector(1., 2., 3., 4.));
}

BOOST_FIXTURE_TEST_CASE(matrix_to_string, MatrixFixture) {
  std::string expected = "1, 2, 3\n2, 3, 4\n3, 4, 5\n4, 5, 6";
  BOOST_TEST(to_string(m43) == expected);
}

struct DiagonalMatrixFixture {
  IdentityMatrix<3> identity{};
  DiagonalMatrix<3> diagonal_identity{1.0, 1.0, 1.0};
  DiagonalMatrix<3> triple_identity{3.0, 3.0, 3.0};
  Matrix<3, 3> square_identity{
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
  };

  DiagonalMatrix<3> diagonal{1.0, 2.0, 3.0};
  Matrix<3> symmetric{
      1.0, 2.0, 2.0, 3.0, 4.0, 3.0,
  };
  Matrix<3, 3> square_asymmetric{
      1.0, 2.0, 4.0, 2.0, 2.0, 4.0, 3.0, 4.0, 3.0,
  };
};

BOOST_FIXTURE_TEST_CASE(diagonal_equality, DiagonalMatrixFixture) {
  BOOST_TEST(diagonal_identity == identity);
  BOOST_TEST(diagonal_identity == square_identity);
  BOOST_TEST(identity == square_identity);
  BOOST_TEST(diagonal != diagonal_identity);
  BOOST_TEST(diagonal != square_asymmetric);
  BOOST_TEST(diagonal != symmetric);
};

BOOST_FIXTURE_TEST_CASE(diagonal_scalar_multiply, DiagonalMatrixFixture) {
  auto r1 = identity * 3.0;
  auto r2 = 3.0 * identity;
  auto r3 = diagonal_identity * 3.0;
  auto r4 = 3.0 * diagonal_identity;
  BOOST_TEST(r1 == triple_identity);
  BOOST_TEST(r2 == triple_identity);
  BOOST_TEST(r3 == triple_identity);
  BOOST_TEST(r4 == triple_identity);
};

BOOST_FIXTURE_TEST_CASE(diagonal_addition, DiagonalMatrixFixture) {
  auto r = diagonal_identity + diagonal_identity;
  auto r1 = r + diagonal_identity;
  auto r2 = identity + identity + identity;
  BOOST_TEST(r1 == triple_identity);
  BOOST_TEST(r2 == triple_identity);
};

BOOST_FIXTURE_TEST_CASE(diagonal_multiplication, DiagonalMatrixFixture) {
  auto r1 = diagonal_identity * diagonal_identity;
  static_assert(
      std::is_same_v<decltype(r1)::result_type, decltype(diagonal_identity)>,
      "Expected result of diagonal multiplication to be diagonal");
  BOOST_TEST(r1 == diagonal_identity);
};

BOOST_AUTO_TEST_CASE(matrix_conversion_test) {
  auto const m = matrix(1, 2, 3, 4);
  static_assert(std::is_same_v<decltype(m[0, 0]), int>,
                "Expected matrix element of m to be int");
  auto const r = m * 0.5f;
  static_assert(std::is_same_v<decltype(r[0, 0]), float>,
                "Expected matrix element of r to be float");
  BOOST_TEST((r[1, 1]) == 2.0);
};

BOOST_AUTO_TEST_CASE(matrix_strassen_test) {
  auto const strassen = Strassen<1>{};
  auto const m1 = matrix(1, 2, 3, 4);
  auto r1 = strassen(m1, m1);
  auto e1 = m1 * m1;
  BOOST_TEST(r1 == e1);

  auto const m2 = matrix(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                         17, 18, 19, 20, 21, 22, 23, 24, 25);
  auto r2 = strassen(m2, m2);
  auto e2 = m2 * m2;
  BOOST_TEST(r2 == e2);
}

ut::test_suite* init_unit_test_suite(int, char*[]) {
  return 0;
}
