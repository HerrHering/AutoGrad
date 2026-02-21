#pragma once

#include <vector>
#include <stdexcept>
#include <format>
#include <cassert>
#include <string>
#include <sstream>

namespace MUtils {
    // Row major matrix
    struct Matrix {
        Matrix(std::vector<float>&& source, int r, int c) : R(r), C(c) {
            // Check dimension
            if (R <= 0 || C <= 0) {
                throw std::invalid_argument(
                    std::format("Matrix can not have negative dimensions. R: {} C: {}", R, C)
                );
            }
            if (R*C != static_cast<int>(source.size())) {
                throw std::invalid_argument(
                    std::format("Matrix size mismatch. Row: {} Col: {} Expected: {} Data: {}", R, C, R*C, source.size())
                );
            }
            // Move data :)
            data = std::move(source);
        }
        Matrix(int r, int c) : data(r*c, 0.0f), R(r), C(c) {}
        inline float& atr(int r, int c) {
            assert(0 <= r && r < R && 0 <= c && c < C && "Matrix access out of bounds.");
            return data.at(r * C + c);
        }
        inline float atc(int r, int c) const {
            assert(0 <= r && r < R && 0 <= c && c < C && "Matrix access out of bounds.");
            return data.at(r * C + c);
        }
        std::vector<float> data;
        const int R, C;
    };

    Matrix mul(const Matrix& A, const Matrix& B) {
        assert(A.C == B.R && "Mismatching matrix dimensions for multiplication!");
        Matrix M{A.R, B.C};

        for (int i = 0; i < A.R; i++) {
            for (int j = 0; j < A.C; j++) { // A.C == B.R
                for (int k = 0; k < B.C; k++) {
                    M.atr(i, k) += A.atc(i, j) * B.atc(j, k);
                }
            }
        }

        return M;
    }

    // A * B^T
    Matrix mult(const Matrix& A, const Matrix& B) {
        assert(A.C == B.C && "Mismatching matrix dimensions for multiplication!");
        Matrix M{A.R, B.C};

        for (int i = 0; i < A.R; i++) {
            for (int k = 0; k < B.R; k++) {
                for (int j = 0; j < A.C; j++) { // A.C == B.C
                    M.atr(i, k) += A.atc(i, j) * B.atc(k,j);
                }
            }
        }

        return M;
    }

    // A^T * B
    Matrix tmul(const Matrix& A, const Matrix& B) {
        assert(A.R == B.R && "Mismatching matrix dimensions for multiplication!");
        Matrix M{A.C, B.C};

        for (int j = 0; j < A.R; j++) { // A.R == B.R
            for (int i = 0; i < A.C; i++) {
                for (int k = 0; k < B.C; k++) {
                    M.atr(i, k) += A.atc(j, i) * B.atc(j, k);
                }
            }
        }

        return M;
    }

    Matrix add(const Matrix& A, const Matrix& B) {
        assert(A.R == B.R && A.C == B.C && "Mismatching matrix dimensions for addition!");
        Matrix M{A.R, A.C};

        for (int i = 0; i < A.R; i++) {
            for (int j = 0; j < A.C; j++) {
                M.atr(i, j) = A.atc(i, j) + B.atc(i, j);
            }
        }

        return M;
    }

    /// @brief Direct product (term wise)
    /// @param A 
    /// @param B 
    /// @return 
    Matrix muld(const Matrix& A, const Matrix& B) {
        assert(A.R == B.R && A.C == B.C && "Mismatching matrix dimensions for direct product!");
        Matrix M{A.R, A.C};

        for (int i = 0; i < A.R; i++) {
            for (int j = 0; j < A.C; j++) {
                M.atr(i, j) = A.atc(i, j) * B.atc(i, j);
            }
        }

        return M;
    }

    /// @brief A - B
    /// @param A 
    /// @param B 
    /// @return 
    Matrix sub(const Matrix& A, const Matrix& B) {
        assert(A.R == B.R && A.C == B.C && "Mismatching matrix dimensions for subtraction!");
        Matrix M{A.R, A.C};

        for (int i = 0; i < A.R; i++) {
            for (int j = 0; j < A.C; j++) {
                M.atr(i, j) = A.atc(i, j) - B.atc(i, j);
            }
        }

        return M;
    }

    Matrix scale(float scale, const Matrix& A) {
        Matrix M{A.R, A.C};

        for (int i = 0; i < A.R; i++) {
            for (int j = 0; j < A.C; j++) {
                M.atr(i, j) = scale * A.atc(i, j);
            }
        }

        return M;
    }

    std::string to_string(const Matrix& m) {
        std::ostringstream oss;
        oss << "\n";
        for (int r = 0; r < m.R; r++) {
            oss << "|";
            for (int c = 0; c < m.C - 1; c++) {
                oss << m.atc(r, c) << ", ";
            }
            oss << m.atc(r, m.C - 1) << "|\n";
        }
        oss << "\n";
        return oss.str();
    }

    Matrix operator*(const Matrix& A, const Matrix& B) { return mul(A, B); }
    Matrix operator*(float s, const Matrix& A) { return scale(s, A); }
    Matrix operator*(const Matrix& A, float s) { return scale(s, A); }
    Matrix operator+(const Matrix& A, const Matrix& B) { return add(A, B); }
    Matrix operator-(const Matrix& A, const Matrix& B) { return sub(A, B); }
}