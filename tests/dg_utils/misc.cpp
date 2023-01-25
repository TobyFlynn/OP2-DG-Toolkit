#include "../catch.hpp"

#include "dg_utils.h"

// Testing DGUtils::numNodes2D
// Basic Constants
TEST_CASE("DGUtils::numNodes2D") {
  SECTION("Order 1") {
    int Np, Nfp;
    DGUtils::numNodes2D(1, &Np, &Nfp);
    REQUIRE(Np == 3);
    REQUIRE(Nfp == 2);
  }
  SECTION("Order 2") {
    int Np, Nfp;
    DGUtils::numNodes2D(2, &Np, &Nfp);
    REQUIRE(Np == 6);
    REQUIRE(Nfp == 3);
  }
  SECTION("Order 3") {
    int Np, Nfp;
    DGUtils::numNodes2D(3, &Np, &Nfp);
    REQUIRE(Np == 10);
    REQUIRE(Nfp == 4);
  }
  SECTION("Order 4") {
    int Np, Nfp;
    DGUtils::numNodes2D(4, &Np, &Nfp);
    REQUIRE(Np == 15);
    REQUIRE(Nfp == 5);
  }
  SECTION("Order 5") {
    int Np, Nfp;
    DGUtils::numNodes2D(5, &Np, &Nfp);
    REQUIRE(Np == 21);
    REQUIRE(Nfp == 6);
  }
  SECTION("Order 12") {
    int Np, Nfp;
    DGUtils::numNodes2D(12, &Np, &Nfp);
    REQUIRE(Np == 91);
    REQUIRE(Nfp == 13);
  }
}

// Testing DG3DUtils::numNodes3D
// Basic Constants
TEST_CASE("DG3DUtils::numNodes3D") {
  SECTION("Order 1") {
    int Np, Nfp;
    DGUtils::numNodes3D(1, &Np, &Nfp);
    REQUIRE(Np == 4);
    REQUIRE(Nfp == 3);
  }
  SECTION("Order 4") {
    int Np, Nfp;
    DGUtils::numNodes3D(4, &Np, &Nfp);
    REQUIRE(Np == 35);
    REQUIRE(Nfp == 15);
  }
  SECTION("Order 12") {
    int Np, Nfp;
    DGUtils::numNodes3D(12, &Np, &Nfp);
    REQUIRE(Np == 455);
    REQUIRE(Nfp == 91);
  }
}