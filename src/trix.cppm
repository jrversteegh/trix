module;

#include <trix/matrix.h>

export module trix;

export namespace trix {
  using trix::Matrix;
  using trix::SymmetricMatrix;
  using trix::DiagonalMatrix;
  using trix::IdentityMatrix;
  using trix::operator==;
  using trix::operator*;
  using trix::operator+;
  using trix::operator<<;
};
