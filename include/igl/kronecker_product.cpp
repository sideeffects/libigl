// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2014 Daniele Panozzo <daniele.panozzo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "kronecker_product.h"

// Bug in unsupported/Eigen/SparseExtra needs iostream first
#include <iostream>
#include <unsupported/Eigen/SparseExtra>

template <typename Scalar>
IGL_INLINE Eigen::SparseMatrix<Scalar> igl::kronecker_product(
  const Eigen::SparseMatrix<Scalar> & A,
  const Eigen::SparseMatrix<Scalar> & B)
{
  using namespace Eigen;
  using namespace std;

  // Convert B in triplets format
  MatrixXd B_triplets(B.nonZeros(),3);
  int count = 0;
  for (int k=0; k<B.outerSize(); ++k)
    for (SparseMatrix<double>::InnerIterator it(B,k); it; ++it)
      B_triplets.row(count++) << it.row(), it.col(), it.value();

  MatrixXd C_triplets(B_triplets.rows()*A.nonZeros(),3);
  count = 0;
  for (int k=0; k<A.outerSize(); ++k)
    for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
    {
      int i = it.row();
      int j = it.col();
      double v = it.value();

      MatrixXd B_triplets_copy = B_triplets;
      B_triplets_copy.col(0) = B_triplets_copy.col(0).array() + double(B.rows()*i);
      B_triplets_copy.col(1) = B_triplets_copy.col(1).array() + double(B.cols()*j);
      B_triplets_copy.col(2) = B_triplets_copy.col(2).array() * v;

      C_triplets.block(count*B_triplets.rows(),0,
                       B_triplets.rows(), B_triplets.cols()) = B_triplets_copy;

      count++;
    }

  typedef Eigen::Triplet<double> T;
  std::vector<T> triplets;
  triplets.reserve(C_triplets.rows());
  
  for(unsigned i=0; i<C_triplets.rows(); ++i)
    triplets.push_back(T(C_triplets(i,0),C_triplets(i,1),C_triplets(i,2)));
  SparseMatrix<Scalar> C(A.rows()*B.rows(),A.cols()*B.cols());
  C.setFromTriplets(triplets.begin(),triplets.end());
  
  return C;
}

#ifndef IGL_HEADER_ONLY
// Explicit template specialization
// generated by autoexplicit.sh
#endif