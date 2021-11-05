#include "../catch.hpp"

#include "dg_utils.h"

#include <string>

static std::string test_data_prefix = "../../tests/dg_utils/data/nodes/";

static void compare_vec(arma::vec &calc, arma::vec &ans) {
  REQUIRE(calc.n_elem == ans.n_elem);

  for(int i = 0; i < calc.n_elem; i++) {
    REQUIRE(calc[i] == Approx(ans[i]).margin(1e-12));
  }
}

// Testing DGUtils::warpFactor
// Warp Factor
TEST_CASE("DGUtils::warpFactor") {
  std::string data_prefix = test_data_prefix + "warpFactor/";

  SECTION("N = 1") {
    int N = 1;
    arma::vec x, ans;
    x.load(data_prefix + "in-N-1.txt");
    ans.load(data_prefix + "out-N-1.txt");
    arma::vec warp = DGUtils::warpFactor(x, N);
    compare_vec(warp, ans);
  }

  SECTION("N = 2") {
    int N = 2;
    arma::vec x, ans;
    x.load(data_prefix + "in-N-2.txt");
    ans.load(data_prefix + "out-N-2.txt");
    arma::vec warp = DGUtils::warpFactor(x, N);
    compare_vec(warp, ans);
  }

  SECTION("N = 4") {
    int N = 4;
    arma::vec x, ans;
    x.load(data_prefix + "in-N-4.txt");
    ans.load(data_prefix + "out-N-4.txt");
    arma::vec warp = DGUtils::warpFactor(x, N);
    compare_vec(warp, ans);
  }

  SECTION("N = 5") {
    int N = 5;
    arma::vec x, ans;
    x.load(data_prefix + "in-N-5.txt");
    ans.load(data_prefix + "out-N-5.txt");
    arma::vec warp = DGUtils::warpFactor(x, N);
    compare_vec(warp, ans);
  }
}

// Testing DGUtils::setRefXY
// X and Y coordinates of nodes on reference element
TEST_CASE("DGUtils::setRefXY") {
  std::string data_prefix = test_data_prefix + "setRefXY/";

  SECTION("N = 1") {
    int N = 1; arma::vec x, y;
    DGUtils::setRefXY(N, x, y);
    arma::vec x_ans, y_ans;
    x_ans.load(data_prefix + "x-N-1.txt");
    y_ans.load(data_prefix + "y-N-1.txt");
    compare_vec(x, x_ans);
    compare_vec(y, y_ans);
  }

  SECTION("N = 2") {
    int N = 2; arma::vec x, y;
    DGUtils::setRefXY(N, x, y);
    arma::vec x_ans, y_ans;
    x_ans.load(data_prefix + "x-N-2.txt");
    y_ans.load(data_prefix + "y-N-2.txt");
    compare_vec(x, x_ans);
    compare_vec(y, y_ans);
  }

  SECTION("N = 5") {
    int N = 5; arma::vec x, y;
    DGUtils::setRefXY(N, x, y);
    arma::vec x_ans, y_ans;
    x_ans.load(data_prefix + "x-N-5.txt");
    y_ans.load(data_prefix + "y-N-5.txt");
    compare_vec(x, x_ans);
    compare_vec(y, y_ans);
  }

  SECTION("N = 8") {
    int N = 8; arma::vec x, y;
    DGUtils::setRefXY(N, x, y);
    arma::vec x_ans, y_ans;
    x_ans.load(data_prefix + "x-N-8.txt");
    y_ans.load(data_prefix + "y-N-8.txt");
    compare_vec(x, x_ans);
    compare_vec(y, y_ans);
  }
}

// Testing DGUtils::xy2rs
// Converting from x-y in an equilateral triangle to r-s in ref triagnle
TEST_CASE("DGUtils::xy2rs") {
  std::string data_prefix = test_data_prefix + "xy2rs/";
  arma::vec x, y, r, s, r_ans, s_ans;
  x.load(data_prefix + "x.txt");
  y.load(data_prefix + "y.txt");
  r_ans.load(data_prefix + "r.txt");
  s_ans.load(data_prefix + "s.txt");
  DGUtils::xy2rs(x, y, r, s);
  compare_vec(r, r_ans);
  compare_vec(s, s_ans);
}

// Testing DGUtils::rs2ab
// Converting from r-s to a-b coordinates
TEST_CASE("DGUtils::rs2ab") {
  std::string data_prefix = test_data_prefix + "rs2ab/";
  arma::vec r, s, a, b, a_ans, b_ans;
  r.load(data_prefix + "r.txt");
  s.load(data_prefix + "s.txt");
  a_ans.load(data_prefix + "a.txt");
  b_ans.load(data_prefix + "b.txt");
  DGUtils::rs2ab(r, s, a, b);
  compare_vec(a, a_ans);
  compare_vec(b, b_ans);
}
