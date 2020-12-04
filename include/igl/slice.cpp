// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "slice.h"
#include "colon.h"

#include <vector>
#include <unsupported/Eigen/SparseExtra>

namespace igl_detail
{
    template <
        typename TX,
        typename DerivedC>
    struct ColNNZ
    {
	typedef int value_type;
	ColNNZ(
		const Eigen::SparseMatrix<TX>& M,
                const Eigen::DenseBase<DerivedC> &C)
	    : myM(M)
	    , myC(C)
	{
	}
	int operator[](int c) const
	{
	    return myM.col(myC(c)).nonZeros();
	}
	const Eigen::SparseMatrix<TX>& myM;
	const Eigen::DenseBase<DerivedC>& myC;
    };
}

template <
    typename TX,
    typename TY,
    typename DerivedR,
    typename DerivedC>
IGL_INLINE void igl::slice(
    const Eigen::SparseMatrix<TX> &X,
    const Eigen::DenseBase<DerivedR> &R,
    const Eigen::DenseBase<DerivedC> &C,
    Eigen::SparseMatrix<TY> &Y)
{
#if 1
  int ym = R.size();
  int yn = C.size();

  // special case when R or C is empty
  if(ym == 0 || yn == 0)
  {
    Y.resize(ym,yn);
    return;
  }

  assert(R.minCoeff() >= 0);
  assert(R.maxCoeff() < X.rows());
  assert(C.minCoeff() >= 0);
  assert(C.maxCoeff() < X.cols());

  // Build reindexing maps for columns and rows (from X to Y)
  std::vector<std::vector<int> > RI;
  RI.resize(X.rows());
  for(int i = 0;i<ym;i++)
  {
    RI[R(i)].push_back(i);
  }
  // Resize output
  Y.resize(ym,yn);
  igl_detail::ColNNZ<TX,DerivedC> col_nnz(X,C);
  Y.reserve(col_nnz);
  for (int c = 0; c < yn; ++c)
  {
    // Iterate over inside
    for(typename Eigen::SparseMatrix<TX>::InnerIterator it(X,C(c)); it; ++it)
    {
      for(int ri : RI[it.row()])
      {
	Y.insert(ri,c) = it.value();
      }
    }
  }
  Y.makeCompressed();
#else
  int xm = X.rows();
  int xn = X.cols();
  int ym = R.size();
  int yn = C.size();

  // special case when R or C is empty
  if (ym == 0 || yn == 0)
  {
    Y.resize(ym, yn);
    return;
  }

  assert(R.minCoeff() >= 0);
  assert(R.maxCoeff() < xm);
  assert(C.minCoeff() >= 0);
  assert(C.maxCoeff() < xn);

  // Build reindexing maps for columns and rows
  std::vector<std::vector<typename DerivedR::Scalar>> RI;
  RI.resize(xm);
  for (int i = 0; i < ym; i++)
  {
    RI[R(i)].push_back(i);
  }
  std::vector<std::vector<typename DerivedC::Scalar>> CI;
  CI.resize(xn);
  for (int i = 0; i < yn; i++)
  {
    CI[C(i)].push_back(i);
  }

  // Take a guess at the number of nonzeros (this assumes uniform distribution
  // not banded or heavily diagonal)
  std::vector<Eigen::Triplet<TY>> entries;
  entries.reserve((X.nonZeros()/(X.rows()*X.cols())) * (ym*yn));

  // Iterate over outside
  for (int k = 0; k < X.outerSize(); ++k)
  {
    // Iterate over inside
    for (typename Eigen::SparseMatrix<TX>::InnerIterator it(X, k); it; ++it)
    {
      for (auto rit = RI[it.row()].begin(); rit != RI[it.row()].end(); rit++)
      {
        for (auto cit = CI[it.col()].begin(); cit != CI[it.col()].end(); cit++)
        {
          entries.emplace_back(*rit, *cit, it.value());
        }
      }
    }
  }
  Y.resize(ym, yn);
  Y.setFromTriplets(entries.begin(), entries.end());
#endif
}

template <
    typename TX,
    typename TY,
    typename DerivedR,
    typename DerivedC>
IGL_INLINE void igl::slice_fast(
    const Eigen::SparseMatrix<TX> &X,
    const Eigen::DenseBase<DerivedR> &R,
    const Eigen::DenseBase<DerivedC> &C,
    Eigen::SparseMatrix<TY> &Y)
{
  typedef typename Eigen::SparseMatrix<TX>::StorageIndex Index;

  Index ym = R.size();
  Index yn = C.size();

  // special case when R or C is empty
  if (ym == 0 || yn == 0)
  {
    Y.resize(ym,yn);
    return;
  }

#ifndef NDEBUG
  assert(X.isCompressed());
  for (Index i = 0; i < ym; ++i)
  {
    assert(R(i) >= 0 && R(i) < X.rows());
    assert(i < 1 || R(i) > R(i-1));
  }
#endif

  // Build inverse map of R, -1 means not in map
  std::vector<int> RI(X.rows(), -1);
  for (int i = 0; i < ym; i++)
    RI[R(i)] = i;

  Y.resize(ym, yn);
  Y.resizeNonZeros(X.nonZeros());

  const Index* x_outer = X.outerIndexPtr();
  const Index* x_inner = X.innerIndexPtr();
  const TX*    x_value = X.valuePtr();
  Index*       y_outer = Y.outerIndexPtr();
  Index*       y_inner = Y.innerIndexPtr();
  TY*	       y_value = Y.valuePtr();
  const Index* y_start = y_inner;

  for (Index c = 0; c < yn; ++c)
  {
    *y_outer++ = Index(y_inner - y_start);
    const Index xc = C(c);
    for (Index xi = x_outer[xc], nxi = x_outer[xc+1]; xi < nxi; ++xi)
    {
      const Index yr = RI[x_inner[xi]];
      if (yr < 0)
	continue;
      *y_inner++ = yr;
      *y_value++ = x_value[xi];
    }
  }
  *y_outer = Index(y_inner - y_start);

  Y.resizeNonZeros(*y_outer); // shrink to actual size

  assert(Y.isCompressed());

#if 0 //ndef NDEBUG
  Eigen::SparseMatrix<TY> Y2;
  slice(X, R, C, Y2);
  assert(Y.rows() == Y2.rows() && Y.cols() == Y2.cols());
  assert(Y.nonZeros() == Y2.nonZeros());
  assert(!memcmp(Y.outerIndexPtr(), Y2.outerIndexPtr(),
		 (Y.cols() + 1) * sizeof(Index)));
  assert(!memcmp(Y.innerIndexPtr(), Y2.innerIndexPtr(),
		 Y.nonZeros() * sizeof(Index)));
  assert(!memcmp(Y.valuePtr(), Y2.valuePtr(), Y.nonZeros() * sizeof(TY)));
#endif
}

template <typename MatX, typename DerivedR, typename MatY>
IGL_INLINE void igl::slice(
    const MatX &X,
    const Eigen::DenseBase<DerivedR> &R,
    const int dim,
    MatY &Y)
{
  Eigen::Matrix<typename DerivedR::Scalar, Eigen::Dynamic, 1> C;
  switch (dim)
  {
  case 1:
    // boring base case
    if (X.cols() == 0)
    {
      Y.resize(R.size(), 0);
      return;
    }
    igl::colon(0, X.cols() - 1, C);
    return slice(X, R, C, Y);
  case 2:
    // boring base case
    if (X.rows() == 0)
    {
      Y.resize(0, R.size());
      return;
    }
    igl::colon(0, X.rows() - 1, C);
    return slice(X, C, R, Y);
  default:
    assert(false && "Unsupported dimension");
    return;
  }
}

template <
    typename DerivedX,
    typename DerivedR,
    typename DerivedC,
    typename DerivedY>
IGL_INLINE void igl::slice(
    const Eigen::DenseBase<DerivedX> &X,
    const Eigen::DenseBase<DerivedR> &R,
    const Eigen::DenseBase<DerivedC> &C,
    Eigen::PlainObjectBase<DerivedY> &Y)
{
#ifndef NDEBUG
  int xm = X.rows();
  int xn = X.cols();
#endif
  int ym = R.size();
  int yn = C.size();

  // special case when R or C is empty
  if (ym == 0 || yn == 0)
  {
    Y.resize(ym, yn);
    return;
  }

  assert(R.minCoeff() >= 0);
  assert(R.maxCoeff() < xm);
  assert(C.minCoeff() >= 0);
  assert(C.maxCoeff() < xn);

  // Resize output
  Y.resize(ym, yn);
  // loop over output rows, then columns
  for (int i = 0; i < ym; i++)
  {
    for (int j = 0; j < yn; j++)
    {
      Y(i, j) = X(R(i, 0), C(j, 0));
    }
  }
}

template <typename DerivedX, typename DerivedY, typename DerivedR>
IGL_INLINE void igl::slice(
    const Eigen::DenseBase<DerivedX> &X,
    const Eigen::DenseBase<DerivedR> &R,
    Eigen::PlainObjectBase<DerivedY> &Y)
{
  // phony column indices
  Eigen::Matrix<typename DerivedR::Scalar, Eigen::Dynamic, 1> C;
  C.resize(1);
  C(0) = 0;
  return igl::slice(X, R, C, Y);
}

template <typename DerivedX, typename DerivedR>
IGL_INLINE DerivedX igl::slice(
    const Eigen::DenseBase<DerivedX> &X,
    const Eigen::DenseBase<DerivedR> &R)
{
  DerivedX Y;
  igl::slice(X, R, Y);
  return Y;
}

template <typename DerivedX, typename DerivedR>
IGL_INLINE DerivedX igl::slice(
    const Eigen::DenseBase<DerivedX> &X,
    const Eigen::DenseBase<DerivedR> &R,
    const int dim)
{
  DerivedX Y;
  igl::slice(X, R, dim, Y);
  return Y;
}

#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
// generated by autoexplicit.sh
template void igl::slice<Eigen::Matrix<double, -1, 2, 0, -1, 2>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::DenseBase<Eigen::Matrix<double, -1, 2, 0, -1, 2> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template Eigen::Matrix<double, -1, -1, 0, -1, -1> igl::slice<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>>(Eigen::DenseBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, int);
template Eigen::Matrix<int, -1, -1, 0, -1, -1> igl::slice<Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>>(Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &);
template Eigen::Matrix<int, -1, 1, 0, -1, 1> igl::slice<Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int);
template void igl::slice<Eigen::Array<bool, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Array<bool, -1, 1, 0, -1, 1> >(Eigen::Array<bool, -1, 1, 0, -1, 1> const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::Array<bool, -1, 1, 0, -1, 1>&);
template void igl::slice<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>>(Eigen::DenseBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> &);
template void igl::slice<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>>(Eigen::DenseBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> &);
template void igl::slice<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, int, Eigen::Matrix<double, -1, -1, 0, -1, -1> &);
template void igl::slice<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>>(Eigen::DenseBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> &);
template void igl::slice<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>>>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, int, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> &);
template void igl::slice<Eigen::Matrix<double, -1, -1, 1, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 1, -1, -1>>(Eigen::DenseBase<Eigen::Matrix<double, -1, -1, 1, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 1, -1, -1>> &);
template void igl::slice<Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>>(Eigen::DenseBase<Eigen::Matrix<double, -1, 1, 0, -1, 1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1>> &);
template void igl::slice<Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::Matrix<double, -1, 1, 0, -1, 1> const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::Matrix<double, -1, 1, 0, -1, 1>&);
template void igl::slice<Eigen::Matrix<double, -1, 3, 0, -1, 3>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 3, 0, -1, 3> >(Eigen::DenseBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> >&);
template void igl::slice<Eigen::Matrix<double, -1, 3, 1, -1, 3>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>>(Eigen::DenseBase<Eigen::Matrix<double, -1, 3, 1, -1, 3>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1>> &);
template void igl::slice<Eigen::Matrix<double, -1, 3, 1, -1, 3>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 3, 1, -1, 3> >(Eigen::Matrix<double, -1, 3, 1, -1, 3> const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::Matrix<double, -1, 3, 1, -1, 3>&);
template void igl::slice<Eigen::Matrix<double, -1, 3, 1, -1, 3>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 3, 1, -1, 3>>(Eigen::DenseBase<Eigen::Matrix<double, -1, 3, 1, -1, 3>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 1, -1, 3>> &);
template void igl::slice<Eigen::Matrix<float, -1, 1, 0, -1, 1>, Eigen::Matrix<float, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>>(Eigen::DenseBase<Eigen::Matrix<float, -1, 1, 0, -1, 1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, 1, 0, -1, 1>> &);
template void igl::slice<Eigen::Matrix<float, -1, 3, 0, -1, 3>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<float, -1, -1, 0, -1, -1>>(Eigen::DenseBase<Eigen::Matrix<float, -1, 3, 0, -1, 3>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, -1, 0, -1, -1>> &);
template void igl::slice<Eigen::Matrix<float, -1, 3, 1, -1, 3>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<float, -1, -1, 0, -1, -1>>(Eigen::DenseBase<Eigen::Matrix<float, -1, 3, 1, -1, 3>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, -1, 0, -1, -1>> &);
template void igl::slice<Eigen::Matrix<float, -1, 3, 1, -1, 3>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<float, -1, 3, 1, -1, 3> >(Eigen::Matrix<float, -1, 3, 1, -1, 3> const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::Matrix<float, -1, 3, 1, -1, 3>&);
template void igl::slice<Eigen::Matrix<float, -1, 3, 1, -1, 3>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<float, -1, 3, 1, -1, 3>>(Eigen::DenseBase<Eigen::Matrix<float, -1, 3, 1, -1, 3>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, 3, 1, -1, 3>> &);
template void igl::slice<Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>>>(Eigen::Matrix<int, -1, -1, 0, -1, -1> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> &);
template void igl::slice<Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Block<Eigen::Matrix<int, -1, -1, 0, -1, -1> const, -1, 1, true>>(Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::DenseBase<Eigen::Block<Eigen::Matrix<int, -1, -1, 0, -1, -1> const, -1, 1, true>> const &, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> &);
template void igl::slice<Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>>(Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> &);
template void igl::slice<Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>>(Eigen::Matrix<int, -1, 1, 0, -1, 1> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, int, Eigen::Matrix<int, -1, 1, 0, -1, 1> &);
template void igl::slice<Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>>>(Eigen::Matrix<int, -1, 1, 0, -1, 1> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> &);
template void igl::slice<Eigen::Matrix<long, -1, 1, 0, -1, 1>, Eigen::Matrix<long, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<long, -1, 1, 0, -1, 1>>>(Eigen::Matrix<long, -1, 1, 0, -1, 1> const &, Eigen::DenseBase<Eigen::Matrix<long, -1, 1, 0, -1, 1>> const &, int, Eigen::PlainObjectBase<Eigen::Matrix<long, -1, 1, 0, -1, 1>> &);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, int, Eigen::Matrix<double, -1, -1, 0, -1, -1>&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::Matrix<double, -1, -1, 0, -1, -1>&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 1, -1, -1> > >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 1, -1, -1> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >, Eigen::Matrix<long, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<long, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 1, -1, -1> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 1, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::Matrix<double, -1, -1, 0, -1, -1>&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 1, -1, 3> > >(Eigen::MatrixBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 1, -1, 3> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<float, -1, -1, 0, -1, -1> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, -1, 0, -1, -1> > >(Eigen::MatrixBase<Eigen::Matrix<float, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, -1, 0, -1, -1> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<float, -1, 3, 0, -1, 3> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, -1, 0, -1, -1> > >(Eigen::MatrixBase<Eigen::Matrix<float, -1, 3, 0, -1, 3> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, -1, 0, -1, -1> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<float, -1, 3, 0, -1, 3> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, 3, 1, -1, 3> > >(Eigen::MatrixBase<Eigen::Matrix<float, -1, 3, 0, -1, 3> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<float, -1, 3, 1, -1, 3> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > >(Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > >(Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >, Eigen::Matrix<long, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > >(Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<long, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
template void igl::slice<Eigen::MatrixBase<Eigen::Matrix<int, -1, 3, 1, -1, 3> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 1, -1, 3> > >(Eigen::MatrixBase<Eigen::Matrix<int, -1, 3, 1, -1, 3> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 1, -1, 3> >&);
template void igl::slice<Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 3, 0, -1, 3> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, int, Eigen::Matrix<double, -1, 3, 0, -1, 3>&);
template void igl::slice<Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>>>(Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> &);
template void igl::slice<Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>>>(Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> &);
template void igl::slice<Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>>, Eigen::Matrix<long, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>>>(Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<long, -1, 1, 0, -1, 1>> const &, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> &);
template void igl::slice<Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 1, -1, 3>>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 1, -1, 3>>>(Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 1, -1, 3>> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 1, -1, 3>> &);
template void igl::slice<Eigen::SparseMatrix<bool, 0, int>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::SparseMatrix<bool, 0, int>>(Eigen::SparseMatrix<bool, 0, int> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, int, Eigen::SparseMatrix<bool, 0, int> &);
template void igl::slice<Eigen::SparseMatrix<double, 0, int>, Eigen::Array<int, -1, 1, 0, -1, 1>, Eigen::SparseMatrix<double, 0, int> >(Eigen::SparseMatrix<double, 0, int> const&, Eigen::DenseBase<Eigen::Array<int, -1, 1, 0, -1, 1> > const&, int, Eigen::SparseMatrix<double, 0, int>&);
template void igl::slice<Eigen::SparseMatrix<double, 0, int>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::SparseMatrix<double, 0, int>>(Eigen::SparseMatrix<double, 0, int> const &, Eigen::DenseBase<Eigen::Matrix<double, -1, 1, 0, -1, 1>> const &, int, Eigen::SparseMatrix<double, 0, int> &);
template void igl::slice<Eigen::SparseMatrix<double, 0, int>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::SparseMatrix<double, 0, int>>(Eigen::SparseMatrix<double, 0, int> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, int, Eigen::SparseMatrix<double, 0, int> &);
template void igl::slice<Eigen::SparseMatrix<double, 0, int>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::SparseMatrix<double, 0, int>>(Eigen::SparseMatrix<double, 0, int> const &, Eigen::DenseBase<Eigen::Matrix<int, -1, 1, 0, -1, 1>> const &, int, Eigen::SparseMatrix<double, 0, int> &);

#ifdef WIN32
template void igl::slice<class Eigen::DenseBase<class Eigen::Matrix<int,-1,-1,0,-1,-1> >,class Eigen::Matrix<__int64,-1,1,0,-1,1>,class Eigen::PlainObjectBase<class Eigen::Matrix<int,-1,-1,0,-1,-1> > >(class Eigen::DenseBase<class Eigen::Matrix<int,-1,-1,0,-1,-1> > const &,class Eigen::DenseBase<class Eigen::Matrix<__int64,-1,1,0,-1,1> > const &,int,class Eigen::PlainObjectBase<class Eigen::Matrix<int,-1,-1,0,-1,-1> > &);
template void igl::slice<class Eigen::MatrixBase<class Eigen::Matrix<int,-1,-1,0,-1,-1> >,class Eigen::Matrix<__int64,-1,1,0,-1,1>,class Eigen::PlainObjectBase<class Eigen::Matrix<int,-1,-1,0,-1,-1> > >(class Eigen::MatrixBase<class Eigen::Matrix<int,-1,-1,0,-1,-1> > const &,class Eigen::DenseBase<class Eigen::Matrix<__int64,-1,1,0,-1,1> > const &,int,class Eigen::PlainObjectBase<class Eigen::Matrix<int,-1,-1,0,-1,-1> > &);
template void igl::slice<Eigen::Matrix<__int64, -1, 1, 0, -1, 1>, Eigen::Matrix<__int64, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<__int64, -1, 1, 0, -1, 1>>>(Eigen::Matrix<__int64, -1, 1, 0, -1, 1> const &, Eigen::DenseBase<Eigen::Matrix<__int64, -1, 1, 0, -1, 1>> const &, int, Eigen::PlainObjectBase<Eigen::Matrix<__int64, -1, 1, 0, -1, 1>> &);
template void igl::slice<Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>>, Eigen::Matrix<__int64, -1, 1, 0, -1, 1>, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>>>(Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> const &, Eigen::DenseBase<Eigen::Matrix<__int64, -1, 1, 0, -1, 1>> const &, int, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1>> &);
#endif

#endif
