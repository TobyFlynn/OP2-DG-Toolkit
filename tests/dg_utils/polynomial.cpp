#include "../catch.hpp"

#include "dg_utils.h"

#include <string>

static std::string test_data_prefix = "../../tests/dg_utils/data/polynomial/";

static void compare_vec(arma::vec &calc, arma::vec &ans) {
  REQUIRE(calc.n_elem == ans.n_elem);

  for(int i = 0; i < calc.n_elem; i++) {
    REQUIRE(calc[i] == Approx(ans[i]).margin(1e-12));
  }
}

// Testing DGUtils::jacobiGQ
// Jacobi Gauss quadature points
TEST_CASE("DGUtils::jacobiGQ") {
  std::string data_prefix = test_data_prefix + "jacobiGQ/";

  SECTION("N = 0 alpha = 1 beta = 1") {
    int N = 0; double alpha = 1.0, beta = 1.0;
    arma::vec x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    arma::vec x_ans, w_ans;
    x_ans.load(data_prefix + "x-N-0-a-1-b-1.txt");
    w_ans.load(data_prefix + "w-N-0-a-1-b-1.txt");
    compare_vec(x, x_ans);
    compare_vec(w, w_ans);
  }

  SECTION("N = 0 alpha = 2 beta = 3") {
    int N = 0; double alpha = 2.0, beta = 3.0;
    arma::vec x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    arma::vec x_ans, w_ans;
    x_ans.load(data_prefix + "x-N-0-a-2-b-3.txt");
    w_ans.load(data_prefix + "w-N-0-a-2-b-3.txt");
    compare_vec(x, x_ans);
    compare_vec(w, w_ans);
  }

  SECTION("N = 3 alpha = 1 beta = 1") {
    int N = 3; double alpha = 1.0, beta = 1.0;
    arma::vec x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    arma::vec x_ans, w_ans;
    x_ans.load(data_prefix + "x-N-3-a-1-b-1.txt");
    w_ans.load(data_prefix + "w-N-3-a-1-b-1.txt");
    compare_vec(x, x_ans);
    compare_vec(w, w_ans);
  }

  SECTION("N = 4 alpha = 1 beta = 1") {
    int N = 4; double alpha = 1.0, beta = 1.0;
    arma::vec x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    arma::vec x_ans, w_ans;
    x_ans.load(data_prefix + "x-N-4-a-1-b-1.txt");
    w_ans.load(data_prefix + "w-N-4-a-1-b-1.txt");
    compare_vec(x, x_ans);
    compare_vec(w, w_ans);
  }

  SECTION("N = 7 alpha = 1 beta = 1") {
    int N = 7; double alpha = 1.0, beta = 1.0;
    arma::vec x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    arma::vec x_ans, w_ans;
    x_ans.load(data_prefix + "x-N-7-a-1-b-1.txt");
    w_ans.load(data_prefix + "w-N-7-a-1-b-1.txt");
    compare_vec(x, x_ans);
    compare_vec(w, w_ans);
  }

  SECTION("N = 7 alpha = 0 beta = 0") {
    int N = 7; double alpha = 0.0, beta = 0.0;
    arma::vec x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    arma::vec x_ans, w_ans;
    x_ans.load(data_prefix + "x-N-7-a-0-b-0.txt");
    w_ans.load(data_prefix + "w-N-7-a-0-b-0.txt");
    compare_vec(x, x_ans);
    compare_vec(w, w_ans);
  }

  SECTION("N = 15 alpha = 0 beta = 0") {
    int N = 15; double alpha = 0.0, beta = 0.0;
    arma::vec x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    arma::vec x_ans, w_ans;
    x_ans.load(data_prefix + "x-N-15-a-0-b-0.txt");
    w_ans.load(data_prefix + "w-N-15-a-0-b-0.txt");
    compare_vec(x, x_ans);
    compare_vec(w, w_ans);
  }
}

// Testing DGUtils::jacobiGL
// Jacobi Gauss Lobatto quadature points
TEST_CASE("DGUtils::jacobiGL") {
  std::string data_prefix = test_data_prefix + "jacobiGL/";

  SECTION("N = 1 alpha = 0 beta = 0") {
    int N = 1; double alpha = 0.0, beta = 0.0;
    arma::vec x = DGUtils::jacobiGL(alpha, beta, N);
    arma::vec x_ans;
    x_ans.load(data_prefix + "x-N-1-a-0-b-0.txt");
    compare_vec(x, x_ans);
  }

  SECTION("N = 2 alpha = 0 beta = 0") {
    int N = 2; double alpha = 0.0, beta = 0.0;
    arma::vec x = DGUtils::jacobiGL(alpha, beta, N);
    arma::vec x_ans;
    x_ans.load(data_prefix + "x-N-2-a-0-b-0.txt");
    compare_vec(x, x_ans);
  }

  SECTION("N = 3 alpha = 0 beta = 0") {
    int N = 3; double alpha = 0.0, beta = 0.0;
    arma::vec x = DGUtils::jacobiGL(alpha, beta, N);
    arma::vec x_ans;
    x_ans.load(data_prefix + "x-N-3-a-0-b-0.txt");
    compare_vec(x, x_ans);
  }

  SECTION("N = 5 alpha = 0 beta = 0") {
    int N = 5; double alpha = 0.0, beta = 0.0;
    arma::vec x = DGUtils::jacobiGL(alpha, beta, N);
    arma::vec x_ans;
    x_ans.load(data_prefix + "x-N-5-a-0-b-0.txt");
    compare_vec(x, x_ans);
  }

  SECTION("N = 8 alpha = 0 beta = 0") {
    int N = 8; double alpha = 0.0, beta = 0.0;
    arma::vec x = DGUtils::jacobiGL(alpha, beta, N);
    arma::vec x_ans;
    x_ans.load(data_prefix + "x-N-8-a-0-b-0.txt");
    compare_vec(x, x_ans);
  }

  SECTION("N = 8 alpha = 2 beta = 1") {
    int N = 8; double alpha = 2.0, beta = 1.0;
    arma::vec x = DGUtils::jacobiGL(alpha, beta, N);
    arma::vec x_ans;
    x_ans.load(data_prefix + "x-N-8-a-2-b-1.txt");
    compare_vec(x, x_ans);
  }
}

// Testing DGUtils::jacobiP
// Jacobi polynomial at specified points
TEST_CASE("DGUtils::jacobiP") {
  std::string data_prefix = test_data_prefix + "jacobiP/";
  arma::vec x;
  x.load(data_prefix + "in.txt");

  SECTION("N = 0 alpha = 0 beta = 0") {
    int N = 0; double alpha = 0.0, beta = 0.0;
    arma::vec res = DGUtils::jacobiP(x, alpha, beta, N);
    arma::vec ans;
    ans.load(data_prefix + "out-N-0-a-0-b-0.txt");
    compare_vec(res, ans);
  }

  SECTION("N = 0 alpha = 1.5 beta = 1") {
    int N = 0; double alpha = 1.5, beta = 1.0;
    arma::vec res = DGUtils::jacobiP(x, alpha, beta, N);
    arma::vec ans;
    ans.load(data_prefix + "out-N-0-a-1_5-b-1.txt");
    compare_vec(res, ans);
  }

  SECTION("N = 1 alpha = 0 beta = 0") {
    int N = 1; double alpha = 0.0, beta = 0.0;
    arma::vec res = DGUtils::jacobiP(x, alpha, beta, N);
    arma::vec ans;
    ans.load(data_prefix + "out-N-1-a-0-b-0.txt");
    compare_vec(res, ans);
  }

  SECTION("N = 2 alpha = 0 beta = 0") {
    int N = 2; double alpha = 0.0, beta = 0.0;
    arma::vec res = DGUtils::jacobiP(x, alpha, beta, N);
    arma::vec ans;
    ans.load(data_prefix + "out-N-2-a-0-b-0.txt");
    compare_vec(res, ans);
  }

  SECTION("N = 2 alpha = 1.5 beta = 1") {
    int N = 2; double alpha = 1.5, beta = 1.0;
    arma::vec res = DGUtils::jacobiP(x, alpha, beta, N);
    arma::vec ans;
    ans.load(data_prefix + "out-N-2-a-1_5-b-1.txt");
    compare_vec(res, ans);
  }

  SECTION("N = 5 alpha = 0 beta = 0") {
    int N = 5; double alpha = 0.0, beta = 0.0;
    arma::vec res = DGUtils::jacobiP(x, alpha, beta, N);
    arma::vec ans;
    ans.load(data_prefix + "out-N-5-a-0-b-0.txt");
    compare_vec(res, ans);
  }

  SECTION("N = 10 alpha = 0 beta = 0") {
    int N = 10; double alpha = 0.0, beta = 0.0;
    arma::vec res = DGUtils::jacobiP(x, alpha, beta, N);
    arma::vec ans;
    ans.load(data_prefix + "out-N-10-a-0-b-0.txt");
    compare_vec(res, ans);
  }
}

// Testing DGUtils::simplex2DP
// 2D orthonomal polynomial on simplex
TEST_CASE("DGUtils::simplex2DP") {
  std::string data_prefix = test_data_prefix + "simplex2DP/";
  arma::vec a, b;
  a.load(data_prefix + "a.txt");
  b.load(data_prefix + "b.txt");

  SECTION("i = 0 j = 0") {
    int i = 0, j = 0;
    arma::vec p = DGUtils::simplex2DP(a, b, i, j);
    arma::vec p_ans;
    p_ans.load(data_prefix + "i-0-j-0.txt");
    compare_vec(p, p_ans);
  }

  SECTION("i = 1 j = 0") {
    int i = 1, j = 0;
    arma::vec p = DGUtils::simplex2DP(a, b, i, j);
    arma::vec p_ans;
    p_ans.load(data_prefix + "i-1-j-0.txt");
    compare_vec(p, p_ans);
  }

  SECTION("i = 0 j = 1") {
    int i = 0, j = 1;
    arma::vec p = DGUtils::simplex2DP(a, b, i, j);
    arma::vec p_ans;
    p_ans.load(data_prefix + "i-0-j-1.txt");
    compare_vec(p, p_ans);
  }

  SECTION("i = 4 j = 6") {
    int i = 4, j = 6;
    arma::vec p = DGUtils::simplex2DP(a, b, i, j);
    arma::vec p_ans;
    p_ans.load(data_prefix + "i-4-j-6.txt");
    compare_vec(p, p_ans);
  }

  SECTION("i = 7 j = 2") {
    int i = 7, j = 2;
    arma::vec p = DGUtils::simplex2DP(a, b, i, j);
    arma::vec p_ans;
    p_ans.load(data_prefix + "i-7-j-2.txt");
    compare_vec(p, p_ans);
  }
}
