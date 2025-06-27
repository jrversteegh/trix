#include "test_common.h"
#include <boost/test/parameterized_test.hpp>

#include "trix/vector.h"

namespace ut = boost::unit_test;

using namespace trix;

struct VectorFixture {
  Vector<3> v3{
    1.0, 2.0, 3.0,
  };
  Vector<4> v4{
    1.0, 2.0, 3.0, 4.0,
  };
  Number emv3{14.0};
  Number emv4{30.0};
  Vector<3> ea3{
    2.0, 4.0, 6.0
  };
  Vector<4> ea4{
    2.0, 4.0, 6.0, 8.0
  };
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
  auto r = v3.str();
  BOOST_TEST(r == "1.0000, 2.0000, 3.0000");
};

ut::test_suite* init_unit_test_suite(int, char*[]) {
/*
UnitQuaternion uqs[] = {
    UnitQuaternion{std::sqrt(0.5f), 0.5f, 0.5f, 0.0f},
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
