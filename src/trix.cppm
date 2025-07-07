module;

#include <trix/vector.h>
#include <trix/matrix.h>
#include <trix/printing.h>

export module trix;

export namespace trix {
  using trix::Vector;
  using trix::Matrix;
  using trix::SymmetricMatrix;
  using trix::DiagonalMatrix;
  using trix::IdentityMatrix;
  using trix::operator==;
  using trix::operator*;
  using trix::operator+;
  using trix::operator<<;
};
