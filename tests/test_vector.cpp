#include "test_common.h"

#include <array>
#include <ranges>
#include <vector>

#include <boost/test/parameterized_test.hpp>

#include "trix/printing.h"
#include "trix/vector.h"

namespace ut = boost::unit_test;
namespace ranges = std::ranges;

using namespace trix;

BOOST_AUTO_TEST_CASE(vector3_construct_from_iter_test) {
  auto source = std::vector{1., 2., 3., 4.};
  auto v1 = Vector<3>(source.begin(), source.end());
  BOOST_TEST(v1 == vector(1., 2., 3.));

  double psource[] = {1., 2., 3.};
  auto v2 = Vector<3>(psource, psource + 3);
  BOOST_TEST(v2 == vector(1., 2., 3.));
}

BOOST_AUTO_TEST_CASE(vector3_construct_from_array_test) {
  auto v1 = Vector<3>(std::array{1., 2., 3.});
  BOOST_TEST(v1 == vector(1., 2., 3.));

  auto source = std::array{1., 2., 3.};
  auto v2 = Vector<3>(source);
  BOOST_TEST(v2 == vector(1., 2., 3.));
}

BOOST_AUTO_TEST_CASE(vector3_construct_from_range_test) {
  auto const to_long = std::array{1., 2., 3., 4.};
  auto v1 = Vector<3>(std::from_range, to_long | ranges::views::take(4));
  BOOST_TEST(v1 == vector(1., 2., 3.));
  v1 = Vector<3>(std::from_range, to_long | ranges::views::take(3));
  BOOST_TEST(v1 == vector(1., 2., 3.));
  v1 = Vector<3>(std::from_range, to_long | ranges::views::take(2));
  BOOST_TEST(v1 == vector(1., 2., 0.));

  auto const to_short = std::array{1., 2.};
  auto v2 = Vector<3>(std::from_range, to_short);
  BOOST_TEST(v2 == vector(1., 2., 0.));
  auto v3 = Vector<3>(std::from_range, ranges::views::iota(1));
  BOOST_TEST(v3 == vector(1., 2., 3.));
}

struct VectorFixture {
  Vector<3> v3{
      1.0,
      2.0,
      3.0,
  };
  Vector<4> v4{
      1.0,
      2.0,
      3.0,
      4.0,
  };
  Number emv3{14.0};
  Number emv4{30.0};
  Number en3{std::sqrt(emv3)};
  Vector<3> ea3{2.0, 4.0, 6.0};
  Vector<4> ea4{2.0, 4.0, 6.0, 8.0};
};

BOOST_FIXTURE_TEST_CASE(vector3_vector3_mul_test, VectorFixture) {
  auto r = v3 * v3;
  BOOST_TEST(r == emv3);
};

BOOST_FIXTURE_TEST_CASE(vector4_vector4_mul_test, VectorFixture) {
  auto r = v4 * v4;
  BOOST_TEST(r == emv4);
};

BOOST_FIXTURE_TEST_CASE(vector3_scalar_mul_test, VectorFixture) {
  auto r1 = v3 * 2.0;
  auto r2 = v3 * 2.0;
  BOOST_TEST(r1.components == 3);
  BOOST_TEST(r2.components == 3);
  BOOST_TEST(r1 == ea3);
  BOOST_TEST(r2 == ea3);
};

BOOST_FIXTURE_TEST_CASE(vector3_vector3_add_test, VectorFixture) {
  auto r = v3 + v3;
  BOOST_TEST(r.components == 3);
  BOOST_TEST(r == ea3);
};

BOOST_FIXTURE_TEST_CASE(vector4_vector4_add_test, VectorFixture) {
  auto r = v4 + v4;
  BOOST_TEST(r.components == 4);
  BOOST_TEST(r == ea4);
};

BOOST_FIXTURE_TEST_CASE(vector3_negation_test, VectorFixture) {
  auto r = -v3;
  auto e = -1. * v3;
  BOOST_TEST(r.components == 3);
  BOOST_TEST(r == e);
};

BOOST_FIXTURE_TEST_CASE(vector3_str_test, VectorFixture) {
  auto r = to_string(v3);
  BOOST_TEST(r == "1, 2, 3");
};

BOOST_FIXTURE_TEST_CASE(vector3_element_test, VectorFixture) {
  BOOST_TEST(v3[0] == 1.);
  v3[0] = 2.;
  BOOST_TEST(v3[0] == 2.);
};

BOOST_FIXTURE_TEST_CASE(vector3_norm_test, VectorFixture) {
  auto r1 = v3.length();
  auto r2 = v3.norm();
  BOOST_TEST(r1 == r2);
  BOOST_TEST(r1 == en3);
};

BOOST_AUTO_TEST_CASE(vector_slice_test) {
  auto v = vector(1., 2., 3., 4.);
  auto s = v.slice<1, 6, 2>();
  BOOST_TEST(s == vector(2., 4.));
  s[1] = 5.;
  BOOST_TEST(s == vector(2., 5.));
  BOOST_TEST(v == vector(1., 2., 3., 5.));
}

BOOST_AUTO_TEST_CASE(vector_const_slice_test) {
  auto const v = vector(1., 2., 3., 4.);
  auto s = v.slice<1, 6, 2>();
  BOOST_TEST(s == vector(2., 4.));
}

ut::test_suite* init_unit_test_suite(int, char*[]) {
  return 0;
}
