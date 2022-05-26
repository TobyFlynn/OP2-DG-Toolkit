#include "dg_utils.h"

#include "cubature_data.h"

#include <cmath>

void DGUtils::jacobiGQ(const double alpha, const double beta, const int N,
                       arma::vec &x, arma::vec &w) {
  x.reset(); w.reset();
  if(N == 0) {
    x.set_size(1); w.set_size(1);
    x[0] = -(alpha - beta) / (alpha + beta + 2.0);
    w[0] = 2.0;
    return;
  }

  arma::vec h1(N + 1);
  for(int i = 0; i < N + 1; i++) {
    h1[i] = 2.0 * i + alpha + beta;
  }

  arma::vec j1(N + 1);
  for(int i = 0; i < N + 1; i++) {
    j1[i] = -0.5 * (alpha * alpha - beta * beta) / (h1[i] + 2.0) / h1[i];
  }

  arma::vec j2(N);
  for(int i = 0; i < N; i++) {
    j2[i]  = 2.0 / (h1[i] + 2.0);
    j2[i] *= sqrt((i + 1.0) * ((i + 1.0) + alpha + beta) * ((i + 1.0) + alpha) *
             ((i + 1.0) + beta) / (h1[i] + 1.0) / (h1[i] + 3.0));
  }
  arma::mat j = diagmat(j1) + diagmat(j2, 1);
  if(alpha + beta < 1e-15)
    j(0,0) = 0.0;

  j = j + j.t();

  arma::vec eigVal;
  arma::mat eigVec;
  arma::eig_sym(eigVal, eigVec, j);

  x.clear(); w.clear();
  x.set_size(eigVal.n_elem); w.set_size(eigVal.n_elem);

  for(int i = 0; i < eigVal.n_elem; i++) {
    x[i] = eigVal(i);
    w[i] = eigVec(0, i) * eigVec(0, i) * pow(2.0, alpha + beta + 1.0) /
           (alpha + beta + 1.0) * tgamma(alpha + 1.0) * tgamma(beta + 1.0) /
           tgamma(alpha + beta + 1.0);
  }
}

arma::vec DGUtils::jacobiGL(const double alpha, const double beta,
                            const int N) {
  arma::vec x(N + 1);
  if(N == 1) {
    x[0] = -1.0;
    x[1] = 1.0;
    return x;
  }

  arma::vec w;
  jacobiGQ(alpha + 1.0, beta + 1.0, N - 2, x, w);
  std::vector<double> x_tmp = arma::conv_to<std::vector<double>>::from(x);
  x_tmp.insert(x_tmp.begin(), -1.0);
  x_tmp.push_back(1.0);
  return arma::vec(x_tmp);
}

// Calculate Jacobi polynomial of order N at points x
arma::vec DGUtils::jacobiP(const arma::vec &x, const double alpha,
                           const double beta, const int N) {
  arma::mat pl(x.n_elem, N + 1);
  double gamma0 = pow(2.0, alpha + beta + 1.0) / (alpha + beta + 1.0) *
                  tgamma(alpha + 1.0) * tgamma(beta + 1.0) /
                  tgamma(alpha + beta + 1.0);
  arma::vec col0(x.n_elem);
  col0.fill(1.0 / sqrt(gamma0));

  // First base case
  if(N == 0) {
    return col0;
  }
  pl.col(0) = col0;

  double gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0;
  pl.col(1) = ((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0) /
              sqrt(gamma1);
  // Second base case
  if(N == 1) {
    return pl.col(1);
  }

  // Recurrence for N > 1
  double aOld = 2.0 / (2.0 + alpha + beta) * sqrt((alpha + 1.0) * (beta + 1.0) /
                (alpha + beta + 3.0));
  for(int i = 1; i < N; i++) {
    double h1 = 2.0 * i + alpha + beta;
    double aNew = 2.0 / (h1 + 2.0) * sqrt((i + 1.0) * (i + 1.0 + alpha + beta) *
                  (i + 1.0 + alpha) * (i + 1.0 + beta) / (h1 + 1.0) /
                  (h1 + 3.0));
    double bNew = -(alpha * alpha - beta * beta) / h1 / (h1 + 2.0);
    pl.col(i + 1) = 1.0 / aNew * (-aOld * pl.col(i - 1) + (x - bNew) %
                    pl.col(i));
    aOld = aNew;
  }

  return pl.col(N);
}

arma::vec DGUtils::gradJacobiP(const arma::vec &x, const double alpha,
                               const double beta, const int N) {
  if(N == 0) {
    return arma::vec(x.n_elem, arma::fill::zeros);
  } else {
    double fact = sqrt(N * (N + alpha + beta + 1.0));
    return fact * jacobiP(x, alpha + 1.0, beta + 1.0, N - 1);
  }
}

arma::vec DGUtils::grad2JacobiP(const arma::vec &x, const double alpha,
                               const double beta, const int N) {
  if(N == 0 || N == 1) {
    return arma::vec(x.n_elem, arma::fill::zeros);
  } else {
    double fact = sqrt(N * (N + alpha + beta + 1.0));
    return fact * gradJacobiP(x, alpha + 1.0, beta + 1.0, N - 1);
  }
}

// Calculate 2D orthonomal poly on simplex of order i,j
arma::vec DGUtils::simplex2DP(const arma::vec &a, const arma::vec &b,
                              const int i, const int j) {
  arma::vec h1 = jacobiP(a, 0, 0, i);
  arma::vec h2 = jacobiP(b, 2 * i + 1, 0, j);
  return sqrt(2.0) * h1 % h2 % arma::pow(1.0 - b, i);
}

// Calculate derivatives of modal basis on simplex
void DGUtils::gradSimplex2DP(const arma::vec &a, const arma::vec &b,
                             const int i, const int j, arma::vec &dr,
                             arma::vec &ds) {
  arma::vec fa  = jacobiP(a, 0.0, 0.0, i);
  arma::vec gb  = jacobiP(b, 2.0 * i + 1.0, 0.0, j);
  arma::vec dfa = gradJacobiP(a, 0.0, 0.0, i);
  arma::vec dgb = gradJacobiP(b, 2.0 * i + 1.0, 0.0, j);

  dr.set_size(arma::size(fa));
  ds.set_size(arma::size(fa));

  // r derivative
  dr = dfa % gb;
  if(i > 0) {
    dr = dr % arma::pow(0.5 * (1.0 - b), i - 1);
  }

  // s derivative
  ds = dfa % (gb % (0.5 * (1.0 + a)));
  if(i > 0) {
    ds = ds % arma::pow(0.5 * (1.0 - b), i - 1);
  }

  arma::vec tmp = dgb % arma::pow(0.5 * (1.0 - b), i);
  if(i > 0) {
    tmp = tmp - 0.5 * i * gb % arma::pow(0.5 * (1.0 - b), i - 1);
  }
  ds = ds + fa % tmp;

  // Normalise
  dr = pow(2.0, i + 0.5) * dr;
  ds = pow(2.0, i + 0.5) * ds;
}

// Calculate simplexes for Hessian
void DGUtils::hessianSimplex2DP(const arma::vec &a, const arma::vec &b,
                                const int i, const int j, arma::vec &dr2,
                                arma::vec &drs, arma::vec &ds2) {
  arma::vec fa   = jacobiP(a, 0.0, 0.0, i);
  arma::vec gb   = jacobiP(b, 2.0 * i + 1.0, 0.0, j);
  arma::vec dfa  = gradJacobiP(a, 0.0, 0.0, i);
  arma::vec dgb  = gradJacobiP(b, 2.0 * i + 1.0, 0.0, j);
  arma::vec dfa2 = grad2JacobiP(a, 0.0, 0.0, i);
  arma::vec dgb2 = grad2JacobiP(b, 2.0 * i + 1.0, 0.0, j);

  // dr2
  dr2 = dfa2 % gb;
  if(i > 1) {
    dr2 = 4.0 * dr2 % arma::pow(1.0 - b, i - 2);
  }

  // dsr
  // dsr = dfa % gb;
  // if(i > 1) {
  //   dsr = 2.0 * dsr % arma::pow(1.0 - b, i - 2);
  // }
  //
  // arma::vec tmp = arma::vec(a.n_elem, arma::fill::zeros);
  // if(i > 0) {
  //   tmp = dgb % arma::pow(1.0 - b, i - 1);
  // }
  // if(i > 1) {
  //   tmp = tmp - i * gb % arma::pow(1.0 - b, i - 2);
  // }
  // tmp = tmp % dfa * 2.0;
  // dsr = dsr + tmp;

  drs = 2.0 * (1.0 + a) % dfa2 % gb;
  if(i > 1) {
    drs = drs % arma::pow(1.0 - b, i - 2);
  }
  arma::vec tmp = dfa % gb * 2.0;
  if(i > 1) {
    tmp = tmp % arma::pow(1.0 - b, i - 2);
  }
  drs = drs + tmp;
  tmp = dgb;
  if(i > 0) {
    tmp = tmp % arma::pow(1.0 - b, i - 1);
    if(i > 1) {
      tmp = tmp - i * gb % arma::pow(1.0 - b, i - 2);
    } else {
      tmp = tmp - i * gb;
    }
  }
  tmp = tmp % dfa * 2.0;
  drs = drs + tmp;

  // ds2
  ds2 = dfa2 % gb % arma::pow(1.0 + a, 2);
  if(i > 1) {
    ds2 = ds2 % arma::pow(1.0 - b, i - 2);
  }

  arma::vec tmp2 = dgb2 % arma::pow(1.0 - b, i);
  if(i > 0) {
    tmp2 = tmp2 - 2.0 * i * dgb % arma::pow(1.0 - b, i - 1);
  }
  if(i > 1) {
    tmp2 = tmp2 + i * (i - 1) * gb % arma::pow(1.0 - b, i - 2);
  }
  ds2 = ds2 + fa % tmp2;

  dr2 = pow(2.0, 0.5) * dr2;
  drs = pow(2.0, 0.5) * drs;
  ds2 = pow(2.0, 0.5) * ds2;
}
// Get cubature rules
void DGUtils::cubature2D(const int cOrder, arma::vec &r, arma::vec &s,
                         arma::vec &w) {
  if(cOrder <= 28) {
    switch(cOrder) {
      case 1:
        r = cubR_1;
        s = cubS_1;
        w = cubW_1;
        break;
      case 2:
        r = cubR_2;
        s = cubS_2;
        w = cubW_2;
        break;
      case 3:
        r = cubR_3;
        s = cubS_3;
        w = cubW_3;
        break;
      case 4:
        r = cubR_4;
        s = cubS_4;
        w = cubW_4;
        break;
      case 5:
        r = cubR_5;
        s = cubS_5;
        w = cubW_5;
        break;
      case 6:
        r = cubR_6;
        s = cubS_6;
        w = cubW_6;
        break;
      case 7:
        r = cubR_7;
        s = cubS_7;
        w = cubW_7;
        break;
      case 8:
        r = cubR_8;
        s = cubS_8;
        w = cubW_8;
        break;
      case 9:
        r = cubR_9;
        s = cubS_9;
        w = cubW_9;
        break;
      case 10:
        r = cubR_10;
        s = cubS_10;
        w = cubW_10;
        break;
      case 11:
        r = cubR_11;
        s = cubS_11;
        w = cubW_11;
        break;
      case 12:
        r = cubR_12;
        s = cubS_12;
        w = cubW_12;
        break;
      case 13:
        r = cubR_13;
        s = cubS_13;
        w = cubW_13;
        break;
      case 14:
        r = cubR_14;
        s = cubS_14;
        w = cubW_14;
        break;
      case 15:
        r = cubR_15;
        s = cubS_15;
        w = cubW_15;
        break;
      case 16:
        r = cubR_16;
        s = cubS_16;
        w = cubW_16;
        break;
      case 17:
        r = cubR_17;
        s = cubS_17;
        w = cubW_17;
        break;
      case 18:
        r = cubR_18;
        s = cubS_18;
        w = cubW_18;
        break;
      case 19:
        r = cubR_19;
        s = cubS_19;
        w = cubW_19;
        break;
      case 20:
        r = cubR_20;
        s = cubS_20;
        w = cubW_20;
        break;
      case 21:
        r = cubR_21;
        s = cubS_21;
        w = cubW_21;
        break;
      case 22:
        r = cubR_22;
        s = cubS_22;
        w = cubW_22;
        break;
      case 23:
        r = cubR_23;
        s = cubS_23;
        w = cubW_23;
        break;
      case 24:
        r = cubR_24;
        s = cubS_24;
        w = cubW_24;
        break;
      case 25:
        r = cubR_25;
        s = cubS_25;
        w = cubW_25;
        break;
      case 26:
        r = cubR_26;
        s = cubS_26;
        w = cubW_26;
        break;
      case 27:
        r = cubR_27;
        s = cubS_27;
        w = cubW_27;
        break;
      case 28:
        r = cubR_28;
        s = cubS_28;
        w = cubW_28;
        break;
    }
  } else {
    int cubN = ceil((cOrder + 1.0) / 2.0);
    arma::vec cubA, cubWA, cubB, cubWB;
    jacobiGQ(0.0, 0.0, cubN - 1, cubA, cubWA);
    jacobiGQ(1.0, 0.0, cubN - 1, cubB, cubWB);

    arma::mat cubAMat = arma::ones(cubN, 1) * cubA.t();
    arma::mat cubBMat = cubB * arma::ones(1, cubN);

    arma::mat cubRMat = 0.5 * (1.0 + cubAMat) % (1.0 - cubBMat) - 1.0;
    arma::mat cubSMat = cubBMat;
    arma::mat cubWMat = 0.5 * cubWB * cubWA.t();

    r = arma::vectorise(cubRMat);
    s = arma::vectorise(cubSMat);
    w = arma::vectorise(cubWMat);
  }
}
