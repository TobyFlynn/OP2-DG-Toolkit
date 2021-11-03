#include "../catch.hpp"

#include "dg_utils.h"

void compare_ans(std::vector<double> &calc, std::vector<double> &ans) {
  REQUIRE(calc.size() == ans.size());

  for(int i = 0; i < calc.size(); i++) {
    REQUIRE(calc[i] == Approx(ans[i]).margin(1e-12));
  }
}

// Testing DGUtils::jacobiGQ
TEST_CASE("Jacobi Gauss quadature points") {
  SECTION("N = 0 alpha = 1 beta = 1") {
    int N = 0; double alpha = 1.0, beta = 1.0;
    std::vector<double> x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    std::vector<double> x_ans{ 0.0 };
    std::vector<double> w_ans{ 2.0 };
    compare_ans(x, x_ans);
    compare_ans(w, w_ans);
  }

  SECTION("N = 0 alpha = 2 beta = 3") {
    int N = 0; double alpha = 2.0, beta = 3.0;
    std::vector<double> x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    std::vector<double> x_ans{ 0.142857142857143 };
    std::vector<double> w_ans{ 2.0 };
    compare_ans(x, x_ans);
    compare_ans(w, w_ans);
  }

  SECTION("N = 3 alpha = 1 beta = 1") {
    int N = 3; double alpha = 1.0, beta = 1.0;
    std::vector<double> x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    std::vector<double> x_ans{ -0.765055323929465, -0.285231516480645, 0.285231516480645, 0.765055323929465 };
    std::vector<double> w_ans{ 0.156949912595694, 0.509716754070973, 0.509716754070973, 0.156949912595694 };
    compare_ans(x, x_ans);
    compare_ans(w, w_ans);
  }

  SECTION("N = 4 alpha = 1 beta = 1") {
    int N = 4; double alpha = 1.0, beta = 1.0;
    std::vector<double> x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    std::vector<double> x_ans{ -0.830223896278567, -0.468848793470714, 4.40241681984739e-17, 0.468848793470714, 0.830223896278567 };
    std::vector<double> w_ans{ 0.0860176821228073, 0.336839460734335, 0.487619047619047, 0.336839460734335, 0.0860176821228077 };
    compare_ans(x, x_ans);
    compare_ans(w, w_ans);
  }

  SECTION("N = 7 alpha = 1 beta = 1") {
    int N = 7; double alpha = 1.0, beta = 1.0;
    std::vector<double> x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    std::vector<double> x_ans{ -0.919533908166459, -0.738773865105505, -0.477924949810445, -0.165278957666387, 0.165278957666387, 0.477924949810444, 0.738773865105505, 0.919533908166459 };
    std::vector<double> w_ans{ 0.020590095649122, 0.102147702360358, 0.225336554969858, 0.318592313687328, 0.318592313687328, 0.225336554969858, 0.102147702360359, 0.0205900956491218 };
    compare_ans(x, x_ans);
    compare_ans(w, w_ans);
  }

  SECTION("N = 7 alpha = 0 beta = 0") {
    int N = 7; double alpha = 0.0, beta = 0.0;
    std::vector<double> x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    std::vector<double> x_ans{ -0.960289856497536, -0.796666477413627, -0.525532409916329, -0.18343464249565, 0.18343464249565, 0.525532409916329, 0.796666477413627, 0.960289856497536 };
    std::vector<double> w_ans{ 0.101228536290376, 0.222381034453374, 0.313706645877888, 0.362683783378362, 0.362683783378361, 0.313706645877888, 0.222381034453374, 0.101228536290376 };
    compare_ans(x, x_ans);
    compare_ans(w, w_ans);
  }

  SECTION("N = 15 alpha = 0 beta = 0") {
    int N = 15; double alpha = 0.0, beta = 0.0;
    std::vector<double> x, w;
    DGUtils::jacobiGQ(alpha, beta, N, x, w);
    std::vector<double> x_ans{ -0.98940093499165, -0.944575023073233, -0.865631202387832, -0.755404408355003, -0.617876244402644, -0.458016777657227, -0.281603550779259, -0.0950125098376373, 0.0950125098376375, 0.281603550779259, 0.458016777657227, 0.617876244402644, 0.755404408355003, 0.865631202387831, 0.944575023073232, 0.98940093499165 };
    std::vector<double> w_ans{ 0.0271524594117538, 0.0622535239386478, 0.0951585116824927, 0.124628971255534, 0.149595988816576, 0.169156519395003, 0.182603415044923, 0.18945061045507, 0.189450610455068, 0.182603415044924, 0.169156519395002, 0.149595988816577, 0.124628971255534, 0.095158511682493, 0.0622535239386468, 0.0271524594117545 };
    compare_ans(x, x_ans);
    compare_ans(w, w_ans);
  }
}

// Testing DGUtils::jacobiGL
TEST_CASE("Jacobi Gauss Lobatto quadature points") {
  SECTION("N = 1 alpha = 0 beta = 0") {
    int N = 1; double alpha = 0.0, beta = 0.0;
    std::vector<double> x = DGUtils::jacobiGL(alpha, beta, N);
    std::vector<double> x_ans{ -1.0, 1.0 };
    compare_ans(x, x_ans);
  }

  SECTION("N = 2 alpha = 0 beta = 0") {
    int N = 2; double alpha = 0.0, beta = 0.0;
    std::vector<double> x = DGUtils::jacobiGL(alpha, beta, N);
    std::vector<double> x_ans{ -1.0, 0.0, 1.0 };
    compare_ans(x, x_ans);
  }

  SECTION("N = 3 alpha = 0 beta = 0") {
    int N = 3; double alpha = 0.0, beta = 0.0;
    std::vector<double> x = DGUtils::jacobiGL(alpha, beta, N);
    std::vector<double> x_ans{ -1.0, -0.447213595499958, 0.447213595499958, 1.0 };
    compare_ans(x, x_ans);
  }

  SECTION("N = 5 alpha = 0 beta = 0") {
    int N = 5; double alpha = 0.0, beta = 0.0;
    std::vector<double> x = DGUtils::jacobiGL(alpha, beta, N);
    std::vector<double> x_ans{ -1.0, -0.765055323929465, -0.285231516480645, 0.285231516480645, 0.765055323929465, 1.0 };
    compare_ans(x, x_ans);
  }

  SECTION("N = 8 alpha = 0 beta = 0") {
    int N = 8; double alpha = 0.0, beta = 0.0;
    std::vector<double> x = DGUtils::jacobiGL(alpha, beta, N);
    std::vector<double> x_ans{ -1.0, -0.89975799541146, -0.677186279510738, -0.363117463826178, -3.00575901416695e-17, 0.363117463826178, 0.677186279510738, 0.89975799541146, 1.0 };
    compare_ans(x, x_ans);
  }

  SECTION("N = 8 alpha = 2 beta = 1") {
    int N = 8; double alpha = 2.0, beta = 1.0;
    std::vector<double> x = DGUtils::jacobiGL(alpha, beta, N);
    std::vector<double> x_ans{ -1.0, -0.867682709669176, -0.657620095997367, -0.382144012218183, -0.0680107502223654, 0.254133137077791, 0.552948698805503, 0.799954679592218, 1.0 };
    compare_ans(x, x_ans);
  }
}
