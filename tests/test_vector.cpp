#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <MUtils.hpp>

using namespace MUtils;

void validate_vector(const Vector& result, const Vector& expected) {
    REQUIRE(result.dim == expected.dim);

    for (size_t i = 0; i < expected.dim; ++i) {
        REQUIRE_THAT(result.atc(i), Catch::Matchers::WithinRel(expected.atc(i), 0.001f));
    }
}

TEST_CASE("Matrix-Vector Multiplication - Basic 2x2", "[vector]") {
    // Setup
    Matrix M{std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, 2, 2};
    Vector v{std::vector<float>{5, 6}};

    validate_vector(mul(M, v), std::vector<float>{17, 39});
}

TEST_CASE("Vector Addition - Basic 2", "[vector]") {
    // Setup
    Vector a{std::vector<float>{1, 2}};
    Vector b{std::vector<float>{3, 4}};

    validate_vector(add(a, b), std::vector<float>{4, 6});
}