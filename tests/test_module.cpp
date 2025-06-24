#include "test_common.h"

import trix;

namespace ut = boost::unit_test;

BOOST_AUTO_TEST_CASE(module_import) {
  trix::Matrix<3, 3> m1{}, m2{};
  BOOST_TEST(m1 == m2);
}

ut::test_suite* init_unit_test_suite(int, char*[]) {
  return 0;
}
