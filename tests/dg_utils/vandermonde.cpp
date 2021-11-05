#include "../catch.hpp"

#include "dg_utils.h"

#include <string>

static std::string test_data_prefix = "../../tests/dg_utils/data/vandermonde/";

static void compare_mat(arma::mat &calc, arma::mat &ans) {
  REQUIRE(calc.n_rows == ans.n_rows);
  REQUIRE(calc.n_cols == ans.n_cols);

  arma::vec calc_vec = arma::vectorise(calc);
  arma::vec ans_vec  = arma::vectorise(ans);
  for(int i = 0; i < calc_vec.size(); i++) {
    REQUIRE(calc_vec[i] == Approx(ans_vec[i]).margin(1e-12));
  }
}

// Testing DGUtils::vandermonde1D
// 1D Vandermonde matrix
TEST_CASE("DGUtils::vandermonde1D") {
  std::string data_prefix = test_data_prefix + "vandermonde1D/";
  arma::vec x;
  x.load(data_prefix + "v1D-in.txt");

  SECTION("N = 1") {
    int N = 1;
    arma::mat v1D = DGUtils::vandermonde1D(x, N);
    arma::mat ans;
    ans.load(data_prefix + "v1D-N-1.txt");
    compare_mat(v1D, ans);
  }

  SECTION("N = 2") {
    int N = 2;
    arma::mat v1D = DGUtils::vandermonde1D(x, N);
    arma::mat ans;
    ans.load(data_prefix + "v1D-N-2.txt");
    compare_mat(v1D, ans);
  }

  SECTION("N = 5") {
    int N = 5;
    arma::mat v1D = DGUtils::vandermonde1D(x, N);
    arma::mat ans;
    ans.load(data_prefix + "v1D-N-5.txt");
    compare_mat(v1D, ans);
  }

  SECTION("N = 10") {
    int N = 10;
    arma::mat v1D = DGUtils::vandermonde1D(x, N);
    arma::mat ans;
    ans.load(data_prefix + "v1D-N-10.txt");
    compare_mat(v1D, ans);
  }
}

// Testing DGUtils::vandermonde2D
// 2D Vandermonde matrix
TEST_CASE("DGUtils::vandermonde2D") {
  std::string data_prefix = test_data_prefix + "vandermonde2D/";

  SECTION("N = 1") {
    int N = 1;
    arma::vec r, s;
    r.load(data_prefix + "r-N-1.txt");
    s.load(data_prefix + "s-N-1.txt");
    arma::mat v2D = DGUtils::vandermonde2D(r, s, N);
    arma::mat v2D_ans;
    v2D_ans.load(data_prefix + "v2D-N-1.txt");
    compare_mat(v2D, v2D_ans);
  }

  SECTION("N = 4") {
    int N = 4;
    arma::vec r, s;
    r.load(data_prefix + "r-N-4.txt");
    s.load(data_prefix + "s-N-4.txt");
    arma::mat v2D = DGUtils::vandermonde2D(r, s, N);
    arma::mat v2D_ans;
    v2D_ans.load(data_prefix + "v2D-N-4.txt");
    compare_mat(v2D, v2D_ans);
  }

  SECTION("N = 7") {
    int N = 7;
    arma::vec r, s;
    r.load(data_prefix + "r-N-7.txt");
    s.load(data_prefix + "s-N-7.txt");
    arma::mat v2D = DGUtils::vandermonde2D(r, s, N);
    arma::mat v2D_ans;
    v2D_ans.load(data_prefix + "v2D-N-7.txt");
    compare_mat(v2D, v2D_ans);
  }
}
