#include "../catch.hpp"

#include "dg_utils.h"

#include <string>

static std::string test_data_prefix = "../../tests/dg_utils/data/matrices/";

static void compare_mat(arma::mat &calc, arma::mat &ans) {
  REQUIRE(calc.n_rows == ans.n_rows);
  REQUIRE(calc.n_cols == ans.n_cols);

  arma::vec calc_vec = arma::vectorise(calc);
  arma::vec ans_vec  = arma::vectorise(ans);
  for(int i = 0; i < calc_vec.size(); i++) {
    REQUIRE(calc_vec[i] == Approx(ans_vec[i]).margin(1e-12));
  }
}

// Testing DGUtils::dMatrices2D
// Calculate differentiation matrices
TEST_CASE("DGUtils::dMatrices2D") {
  std::string data_prefix = test_data_prefix + "dMatrices2D/";

  SECTION("N = 1") {
    int N = 1;
    arma::vec r, s;
    r.load(data_prefix + "r-N-1.txt");
    s.load(data_prefix + "s-N-1.txt");
    arma::mat v, dr, ds, dr_ans, ds_ans;
    v.load(data_prefix + "V-N-1.txt");
    DGUtils::dMatrices2D(r, s, v, N, dr, ds);
    dr_ans.load(data_prefix + "dr-N-1.txt");
    ds_ans.load(data_prefix + "ds-N-1.txt");
    compare_mat(dr, dr_ans);
    compare_mat(ds, ds_ans);
  }

  SECTION("N = 4") {
    int N = 4;
    arma::vec r, s;
    r.load(data_prefix + "r-N-4.txt");
    s.load(data_prefix + "s-N-4.txt");
    arma::mat v, dr, ds, dr_ans, ds_ans;
    v.load(data_prefix + "V-N-4.txt");
    DGUtils::dMatrices2D(r, s, v, N, dr, ds);
    dr_ans.load(data_prefix + "dr-N-4.txt");
    ds_ans.load(data_prefix + "ds-N-4.txt");
    compare_mat(dr, dr_ans);
    compare_mat(ds, ds_ans);
  }

  SECTION("N = 7") {
    int N = 7;
    arma::vec r, s;
    r.load(data_prefix + "r-N-7.txt");
    s.load(data_prefix + "s-N-7.txt");
    arma::mat v, dr, ds, dr_ans, ds_ans;
    v.load(data_prefix + "V-N-7.txt");
    DGUtils::dMatrices2D(r, s, v, N, dr, ds);
    dr_ans.load(data_prefix + "dr-N-7.txt");
    ds_ans.load(data_prefix + "ds-N-7.txt");
    compare_mat(dr, dr_ans);
    compare_mat(ds, ds_ans);
  }
}

// Testing DGUtils::dMatrices3D
// Calculate differentiation matrices
TEST_CASE("DGUtils::dMatrices3D") {
  std::string data_prefix = test_data_prefix + "dMatrices3D/";

  SECTION("N = 1") {
    int N = 1;
    arma::vec r, s, t;
    r.load(data_prefix + "r-N-1.txt");
    s.load(data_prefix + "s-N-1.txt");
    t.load(data_prefix + "t-N-1.txt");
    arma::mat v, dr, ds, dt, dr_ans, ds_ans, dt_ans;
    v.load(data_prefix + "v-N-1.txt");
    DGUtils::dMatrices3D(r, s, t, v, N, dr, ds, dt);
    dr_ans.load(data_prefix + "dr-N-1.txt");
    ds_ans.load(data_prefix + "ds-N-1.txt");
    dt_ans.load(data_prefix + "dt-N-1.txt");
    compare_mat(dr, dr_ans);
    compare_mat(ds, ds_ans);
    compare_mat(dt, dt_ans);
  }

  SECTION("N = 4") {
    int N = 4;
    arma::vec r, s, t;
    r.load(data_prefix + "r-N-4.txt");
    s.load(data_prefix + "s-N-4.txt");
    t.load(data_prefix + "t-N-4.txt");
    arma::mat v, dr, ds, dt, dr_ans, ds_ans, dt_ans;
    v.load(data_prefix + "v-N-4.txt");
    DGUtils::dMatrices3D(r, s, t, v, N, dr, ds, dt);
    dr_ans.load(data_prefix + "dr-N-4.txt");
    ds_ans.load(data_prefix + "ds-N-4.txt");
    dt_ans.load(data_prefix + "dt-N-4.txt");
    compare_mat(dr, dr_ans);
    compare_mat(ds, ds_ans);
    compare_mat(dt, dt_ans);
  }
}

// Testing DGUtils::lift2D
// Surface to volume lift matrix
TEST_CASE("DGUtils::lift2D") {
  std::string data_prefix = test_data_prefix + "lift2D/";

  SECTION("N = 1") {
    int N = 1;
    arma::vec r, s; arma::uvec fmask;
    r.load(data_prefix + "r-N-1.txt");
    s.load(data_prefix + "s-N-1.txt");
    fmask.load(data_prefix + "fmask-N-1.txt");
    arma::mat v, lift_ans;
    v.load(data_prefix + "V-N-1.txt");
    arma::mat lift = DGUtils::lift2D(r, s, fmask, v, N);
    lift_ans.load(data_prefix + "LIFT-N-1.txt");
    compare_mat(lift, lift_ans);
  }

  SECTION("N = 2") {
    int N = 2;
    arma::vec r, s; arma::uvec fmask;
    r.load(data_prefix + "r-N-2.txt");
    s.load(data_prefix + "s-N-2.txt");
    fmask.load(data_prefix + "fmask-N-2.txt");
    arma::mat v, lift_ans;
    v.load(data_prefix + "V-N-2.txt");
    arma::mat lift = DGUtils::lift2D(r, s, fmask, v, N);
    lift_ans.load(data_prefix + "LIFT-N-2.txt");
    compare_mat(lift, lift_ans);
  }

  SECTION("N = 3") {
    int N = 3;
    arma::vec r, s; arma::uvec fmask;
    r.load(data_prefix + "r-N-3.txt");
    s.load(data_prefix + "s-N-3.txt");
    fmask.load(data_prefix + "fmask-N-3.txt");
    arma::mat v, lift_ans;
    v.load(data_prefix + "V-N-3.txt");
    arma::mat lift = DGUtils::lift2D(r, s, fmask, v, N);
    lift_ans.load(data_prefix + "LIFT-N-3.txt");
    compare_mat(lift, lift_ans);
  }

  SECTION("N = 4") {
    int N = 4;
    arma::vec r, s; arma::uvec fmask;
    r.load(data_prefix + "r-N-4.txt");
    s.load(data_prefix + "s-N-4.txt");
    fmask.load(data_prefix + "fmask-N-4.txt");
    arma::mat v, lift_ans;
    v.load(data_prefix + "V-N-4.txt");
    arma::mat lift = DGUtils::lift2D(r, s, fmask, v, N);
    lift_ans.load(data_prefix + "LIFT-N-4.txt");
    compare_mat(lift, lift_ans);
  }

  SECTION("N = 5") {
    int N = 5;
    arma::vec r, s; arma::uvec fmask;
    r.load(data_prefix + "r-N-5.txt");
    s.load(data_prefix + "s-N-5.txt");
    fmask.load(data_prefix + "fmask-N-5.txt");
    arma::mat v, lift_ans;
    v.load(data_prefix + "V-N-5.txt");
    arma::mat lift = DGUtils::lift2D(r, s, fmask, v, N);
    lift_ans.load(data_prefix + "LIFT-N-5.txt");
    compare_mat(lift, lift_ans);
  }
}

// Testing DGUtils::lift3D
// Surface to volume lift matrix
TEST_CASE("DGUtils::lift3D") {
  std::string data_prefix = test_data_prefix + "lift3D/";

  SECTION("N = 1") {
    int N = 1;
    arma::vec r, s, t;
    arma::uvec fmask;
    r.load(data_prefix + "r-N-1.txt");
    s.load(data_prefix + "s-N-1.txt");
    t.load(data_prefix + "t-N-1.txt");
    fmask.load(data_prefix + "fmask-N-1.txt");
    arma::mat v, lift_ans;
    v.load(data_prefix + "v-N-1.txt");
    arma::mat lift = DGUtils::lift3D(r, s, t, fmask, v, N);
    lift_ans.load(data_prefix + "lift-N-1.txt");
    compare_mat(lift, lift_ans);
  }

  SECTION("N = 4") {
    int N = 4;
    arma::vec r, s, t;
    arma::uvec fmask;
    r.load(data_prefix + "r-N-4.txt");
    s.load(data_prefix + "s-N-4.txt");
    t.load(data_prefix + "t-N-4.txt");
    fmask.load(data_prefix + "fmask-N-4.txt");
    arma::mat v, lift_ans;
    v.load(data_prefix + "v-N-4.txt");
    arma::mat lift = DGUtils::lift3D(r, s, t, fmask, v, N);
    lift_ans.load(data_prefix + "lift-N-4.txt");
    compare_mat(lift, lift_ans);
  }
}