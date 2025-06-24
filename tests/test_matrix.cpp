#include "test_common.h"
#include <boost/test/parameterized_test.hpp>

#include "trix/matrix.h"

namespace ut = boost::unit_test;

using namespace trix;


struct MatrixFixture {
  Matrix<3,4> m43{
    1.0, 2.0, 3.0, 4.0,
    2.0, 3.0, 4.0, 5.0,
    3.0, 4.0, 5.0, 6.0,
  };
  Matrix<4,3> m34{
    1.0, 2.0, 3.0,
    2.0, 3.0, 4.0,
    3.0, 4.0, 5.0,
    4.0, 5.0, 6.0,
  };
  Matrix<3,3> e33{
    30.0, 40.0, 50.0,
    40.0, 54.0, 68.0,
    50.0, 68.0, 86.0,
  };
  Matrix<4,4> e44{
    14.0, 20.0, 26.0, 32.0,
    20.0, 29.0, 38.0, 47.0,
    26.0, 38.0, 50.0, 62.0,
    32.0, 47.0, 62.0, 77.0,
  };
};


BOOST_FIXTURE_TEST_CASE(matrix_mul_test3, MatrixFixture) {
  auto r = m43 * m34;
  BOOST_TEST(r.rows == 3);
  BOOST_TEST(r.columns == 3);
  BOOST_TEST(r == e33);
};

BOOST_FIXTURE_TEST_CASE(matrix_mul_test4, MatrixFixture) {
  auto r = m34 * m43;
  BOOST_TEST(r.rows == 4);
  BOOST_TEST(r.columns == 4);
  BOOST_TEST(r == e44);
};

BOOST_FIXTURE_TEST_CASE(matrix_scalar_mul_test, MatrixFixture) {
  auto r1 = 2 * m34;
  auto r2 = m34 * 2;
  auto e = m34 + m34;
  BOOST_TEST(r1 == e);
  BOOST_TEST(r2 == e);
};


struct SymmetricMatrixFixture {

  SymmetricMatrix<3> symmetric {
    1.0,
    2.0, 3.0,
    3.0, 4.0, 5.0,
  };

  Matrix<3, 3> square{symmetric};
  Matrix<3, 3> square_asymmetric{
    1.0, 2.0, 4.0,
    2.0, 3.0, 4.0,
    3.0, 4.0, 5.0,
  };

  SymmetricMatrix<3> symmetric_multiplied{
    14.,
    20., 29.,
    26., 38., 50.,
  };

  Matrix<3, 3> square_multiplied{symmetric_multiplied};

  SymmetricMatrix<3> symmetric_added{2 * symmetric};
  Matrix<3, 3> square_added{symmetric_added};

  SymmetricMatrix<3> symmetric_zero{};
  Matrix<3, 3> square_zero{};
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

template <template<typename, typename> typename OPERATION>
struct OperationTest {
  template <typename ARG1, typename ARG2, typename EXPECTED>
  void operator()(ARG1 const& arg1, ARG2 const& arg2, EXPECTED const& expected) {
    OPERATION<ARG1, ARG2> operation{};
    auto r = operation(arg1, arg2);
    BOOST_TEST((typeid(r) == typeid(expected)));
    BOOST_TEST(r == expected);
  }
};

void do_cross_test(auto& test, auto arg1, auto arg2, auto expected1, auto expected2) {
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
  do_cross_test(test, square, symmetric, square_multiplied, symmetric_multiplied);
};

BOOST_FIXTURE_TEST_CASE(matrix_symmetric_add, SymmetricMatrixFixture) {
  OperationTest<Add> test{};
  do_cross_test(test, square, symmetric, square_added, symmetric_added);
};

BOOST_FIXTURE_TEST_CASE(matrix_symmetric_subtract, SymmetricMatrixFixture) {
  OperationTest<Subtract> test{};
  do_cross_test(test, square, symmetric, square_zero, symmetric_zero);
};

struct DiagonalMatrixFixture {
  IdentityMatrix<3> identity{};
  DiagonalMatrix<3> diagonal_identity{1.0, 1.0, 1.0};
  DiagonalMatrix<3> triple_identity{3.0, 3.0, 3.0};
  Matrix<3, 3> square_identity{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  };

  DiagonalMatrix<3> diagonal{1.0, 2.0, 3.0};
  Matrix<3> symmetric{
    1.0,
    2.0, 2.0,
    3.0, 4.0, 3.0,
  };
  Matrix<3, 3> square_asymmetric{
    1.0, 2.0, 4.0,
    2.0, 2.0, 4.0,
    3.0, 4.0, 3.0,
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
  auto r1 = diagonal_identity + diagonal_identity + diagonal_identity;
  auto r2 = identity + identity + identity;
  BOOST_TEST(r1 == triple_identity);
  BOOST_TEST(r2 == triple_identity);
};

BOOST_FIXTURE_TEST_CASE(diagonal_multiplication, DiagonalMatrixFixture) {
  auto r1 = diagonal_identity * diagonal_identity;
  static_assert(std::is_same_v<decltype(r1), decltype(diagonal_identity)>, "Expected result of diagonal multiplication to be diagonal");
  BOOST_TEST(r1 == diagonal_identity);
};

BOOST_FIXTURE_TEST_CASE(matrix_transpose, MatrixFixture) {
  BOOST_TEST(m43.transpose() == m34);
  BOOST_TEST((m43 * m43.transpose()) == (m43 * m34));
  BOOST_TEST((m34 + m43.transpose()) == (2 * m34));
}


ut::test_suite* init_unit_test_suite(int, char*[]) {
/*
UnitQuaternion uqs[] = {
    UnitQuaternion{std::sqrt(0.5f), 0.5f, 0.5f, 0.0f},
    UnitQuaternion{0.0f, std::sqrt(0.5f), 0.5f, 0.5f},
    UnitQuaternion{0.0f, 0.5f, std::sqrt(0.5f), 0.5f},
    UnitQuaternion{0.5f, 0.0f, 0.5f, std::sqrt(0.5f)},
    UnitQuaternion{std::sqrt(0.5f), -0.5f, 0.5f, 0.0f},
    UnitQuaternion{0.0f, std::sqrt(0.5f), -0.5f, 0.5f},
    UnitQuaternion{0.0f, 0.5f, std::sqrt(0.5f), -0.5f},
    UnitQuaternion{-0.5f, 0.0f, 0.5f, std::sqrt(0.5f)},
    UnitQuaternion{-1.f * std::sqrt(0.5f), 0.5f, 0.5f, 0.0f},
    UnitQuaternion{0.0f, -1.f * std::sqrt(0.5f), 0.5f, 0.5f},
    UnitQuaternion{0.0f, 0.5f, -1.f * std::sqrt(0.5f), 0.5f},
    UnitQuaternion{0.5f, 0.0f, 0.5f, -1.f * std::sqrt(0.5f)}};

void test_qmq_conversion(UnitQuaternion q) {
  RotationMatrix m{q};
  UnitQuaternion result = static_cast<UnitQuaternion>(m);
  BOOST_TEST((result == q || result == -q));
}

*/
//  ut::framework::master_test_suite().add(BOOST_PARAM_TEST_CASE(
//      &test_qmq_conversion, uqs, uqs + sizeof(uqs) / sizeof(uqs[0])));
  return 0;
}
