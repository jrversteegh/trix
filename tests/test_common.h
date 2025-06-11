#include <stdio.h>
#include <string.h>
#include <string>

#include <boost/test/unit_test.hpp>

#define TEST

namespace tt = boost::test_tools;
namespace ut = boost::unit_test;

#ifdef WIN32
#define LINE_END "\r\n"
#else
#define LINE_END "\n"
#endif
