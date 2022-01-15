#include "../catch.hpp"

#include "dg_utils.h"

// Testing DGUtils::basic_constants
// Basic Constants
TEST_CASE("DGUtils::basic_constants") {
  SECTION("Order 1") {
    int Np, Nfp;
    DGUtils::basic_constants(1, &Np, &Nfp);
    REQUIRE(Np == 3);
    REQUIRE(Nfp == 2);
  }
  SECTION("Order 2") {
    int Np, Nfp;
    DGUtils::basic_constants(2, &Np, &Nfp);
    REQUIRE(Np == 6);
    REQUIRE(Nfp == 3);
  }
  SECTION("Order 3") {
    int Np, Nfp;
    DGUtils::basic_constants(3, &Np, &Nfp);
    REQUIRE(Np == 10);
    REQUIRE(Nfp == 4);
  }
  SECTION("Order 4") {
    int Np, Nfp;
    DGUtils::basic_constants(4, &Np, &Nfp);
    REQUIRE(Np == 15);
    REQUIRE(Nfp == 5);
  }
  SECTION("Order 5") {
    int Np, Nfp;
    DGUtils::basic_constants(5, &Np, &Nfp);
    REQUIRE(Np == 21);
    REQUIRE(Nfp == 6);
  }
  SECTION("Order 12") {
    int Np, Nfp;
    DGUtils::basic_constants(12, &Np, &Nfp);
    REQUIRE(Np == 91);
    REQUIRE(Nfp == 13);
  }
}
