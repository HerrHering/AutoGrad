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
            static_assert(std::is_move_assignable_v<Matrix>);
            static_assert(std::is_move_constructible_v<Matrix>);
            static_assert(std::is_copy_assignable_v<Matrix>);
            static_assert(std::is_copy_constructible_v<Matrix>);
            static_assert(!std::is_default_constructible_v<Matrix>);
            // Check dimension
#ifndef NDEBUG
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
#endif
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
        int R, C;
    };

    /// @brief Column vector
    struct Vector {
        Vector(std::vector<float>&& source) : dim(static_cast<int>(source.size())), data{std::move(source), static_cast<int>(source.size()), 1} {
            static_assert(std::is_move_assignable_v<Vector>);
            static_assert(std::is_move_constructible_v<Vector>);
            static_assert(std::is_copy_assignable_v<Vector>);
            static_assert(std::is_copy_constructible_v<Vector>);
            static_assert(!std::is_default_constructible_v<Vector>);
        }
        explicit Vector(Matrix&& vector_as_matrix) : Vector(std::move(vector_as_matrix.data)) {}
        Vector(int dim) : dim(dim), data(dim, 1) {}
        inline float& atr(int i) {
            return data.atr(i, 0);
        }
        inline float atc(int i) const {
            return data.atc(i, 0);
        }
        int dim;
        Matrix data;
    };

    inline Matrix mul(const Matrix& A, const Matrix& B) {
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
    inline Matrix mult(const Matrix& A, const Matrix& B) {
        assert(A.C == B.C && "Mismatching matrix dimensions for multiplication!");
        Matrix M{A.R, B.R};

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
    inline Matrix tmul(const Matrix& A, const Matrix& B) {
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

    inline Matrix add(const Matrix& A, const Matrix& B) {
        assert(A.R == B.R && A.C == B.C && "Mismatching matrix dimensions for addition!");
        Matrix M{A.R, A.C};

        for (int i = 0; i < (int)A.data.size(); i++) {
            M.data[i] = A.data[i] + B.data[i];
        }

        return M;
    }

    /// @brief Direct product (term wise)
    /// @param A 
    /// @param B 
    /// @return 
    inline Matrix muld(const Matrix& A, const Matrix& B) {
        assert(A.R == B.R && A.C == B.C && "Mismatching matrix dimensions for direct product!");
        Matrix M{A.R, A.C};

        for (int i = 0; i < (int)A.data.size(); i++) {
            M.data[i] = A.data[i] * B.data[i];
        }

        return M;
    }

    /// @brief A - B
    /// @param A 
    /// @param B 
    /// @return 
    inline Matrix sub(const Matrix& A, const Matrix& B) {
        assert(A.R == B.R && A.C == B.C && "Mismatching matrix dimensions for subtraction!");
        Matrix M{A.R, A.C};

        for (int i = 0; i < (int)A.data.size(); i++) {
            M.data[i] = A.data[i] - B.data[i];
        }

        return M;
    }

    inline Matrix scale(float scale, const Matrix& A) {
        Matrix M{A.R, A.C};

        for (int i = 0; i < (int)A.data.size(); i++) {
            M.data[i] = scale * A.data[i];
        }

        return M;
    }

    inline std::string to_string(const Matrix& m) {
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

    inline Matrix operator*(const Matrix& A, const Matrix& B) { return mul(A, B); }
    inline Matrix operator*(float s, const Matrix& A) { return scale(s, A); }
    inline Matrix operator*(const Matrix& A, float s) { return scale(s, A); }
    inline Matrix operator+(const Matrix& A, const Matrix& B) { return add(A, B); }
    inline Matrix operator-(const Matrix& A, const Matrix& B) { return sub(A, B); }

    /// @brief A*v
    inline Vector mul(const Matrix& A, const Vector& v) {
        assert(A.C == v.dim && "Mismatching matrix-vector dimensions for multiplication!");
        Vector R{A.R};

        for (int i = 0; i < A.R; i++) {
            for (int j = 0; j < v.dim; j++) { // A.C == v.dim
                R.atr(i) += A.atc(i, j) * v.atc(j);
            }
        }

        return R;
    }

    /// @brief Outer product of two vectors a*b^T 
    inline Matrix mulo(const Vector& a, const Vector& b) {
        Matrix M{a.dim, b.dim};

        for (int i = 0; i < a.dim; i++) {
            for (int j = 0; j < b.dim; j++) {
                M.atr(i, j) = a.atc(i) * b.atc(j);
            }
        }

        return M;
    }

    inline Vector add(const Vector& a, const Vector& b) {
        return Vector(add(a.data, b.data));
    }
    inline Vector sub(const Vector& a, const Vector& b) {
        return Vector(sub(a.data, b.data));
    }

    inline Vector muld(const Vector& a, const Vector& b) {
        return Vector(muld(a.data, b.data));
    }
    inline Vector scale(float scaler, const Vector& v) {
        return Vector(scale(scaler, v.data));
    }

    inline Vector operator+(const Vector& a, const Vector& b) { return add(a,b); }
    inline Vector operator-(const Vector& a, const Vector& b) { return sub(a, b); }
    inline Vector operator*(const Matrix& M, const Vector& v) { return mul(M,v); }
    inline Vector operator*(float scaler, const Vector& v) { return scale(scaler,v); }
}