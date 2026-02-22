#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <MUtils.hpp>

using namespace MUtils;

void validate_matrix(const Matrix& result, const Matrix& expected) {
    REQUIRE(result.R == expected.R);
    REQUIRE(result.C == expected.C);
    REQUIRE(result.data.size() == expected.data.size());

    for (size_t i = 0; i < expected.data.size(); ++i) {
        // INFO prints the index only if the test fails, helping you debug
        //INFO("Error at index " << i); 
        REQUIRE_THAT(result.data.at(i), Catch::Matchers::WithinRel(expected.data.at(i), 0.001f));
    }
}

TEST_CASE("Matrix Multiplication - Basic 2x2", "[matrix]") {
    // Setup
    // Assuming your Matrix class looks something like Matrix(rows, cols)
    Matrix A{{1.0f, 2.0f, 3.0f, 4.0f}, 2, 2};
    Matrix B{{5.0f, 6.0f, 7.0f, 8.0f}, 2, 2};

    SECTION("mul()") {
        validate_matrix(mul(A, B), Matrix{std::vector<float>{19, 22, 43, 50}, 2, 2});
    }
    SECTION("tmul()") {
        validate_matrix(tmul(A, B), Matrix{std::vector<float>{26, 30, 38, 44}, 2, 2});
    }
    SECTION("mult()") {
        validate_matrix(mult(A, B), Matrix{std::vector<float>{17, 23, 39, 53}, 2, 2});
    }
}

TEST_CASE("Matrix Addition - Basic 2x2", "[matrix]") {
    // Setup
    // Assuming your Matrix class looks something like Matrix(rows, cols)
    Matrix A{{1.0f, 2.0f, 3.0f, 4.0f}, 2, 2};
    Matrix B{{5.0f, 6.0f, 7.0f, 8.0f}, 2, 2};

    validate_matrix(add(A, B), Matrix{std::vector<float>{6, 8, 10, 12}, 2, 2});
}