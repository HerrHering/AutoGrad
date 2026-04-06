#pragma once

#include <vector>
#include <stdexcept>
#include <format>
#include <cassert>
#include <string>
#include <sstream>
#include <Eigen/Dense>

namespace MUtils {
    // Wrapper around Eigen::MatrixXf for optimized linear algebra
    struct Matrix {
        // Internal Eigen matrix (column-major, SIMD-optimized) - declared first for initialization order
        Eigen::MatrixXf mat;
        int R, C;  // rows and cols
        
        // Constructors
        Matrix(std::vector<float>&& source, int r, int c) : mat(r, c), R(r), C(c) {
            // Check dimension
#ifndef NDEBUG
            if (r <= 0 || c <= 0) {
                throw std::invalid_argument(
                    std::format("Matrix can not have negative dimensions. R: {} C: {}", r, c)
                );
            }
            if (r*c != static_cast<int>(source.size())) {
                throw std::invalid_argument(
                    std::format("Matrix size mismatch. Row: {} Col: {} Expected: {} Data: {}", r, c, r*c, source.size())
                );
            }
#endif
            // Load data into Eigen matrix (convert from row-major to Eigen's column-major)
            mat = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                source.data(), r, c
            ).eval();  // Evaluate to store in Eigen's column-major format
        }
        
        Matrix(int r, int c) : mat(r, c), R(r), C(c) {
            mat.setZero();
        }
        
        // Default constructor
        Matrix() : mat(1, 1), R(1), C(1) { }
        
        // Default move and copy semantics
        Matrix(const Matrix&) = default;
        Matrix(Matrix&&) = default;
        Matrix& operator=(const Matrix&) = default;
        Matrix& operator=(Matrix&&) = default;
        
        // Access methods
        inline float& atr(int r, int c) {
            assert(0 <= r && r < R && 0 <= c && c < C && "Matrix access out of bounds.");
            return mat(r, c);
        }
        
        inline float atc(int r, int c) const {
            assert(0 <= r && r < R && 0 <= c && c < C && "Matrix access out of bounds.");
            return mat(r, c);
        }
        
        // Iterator interface for data access
        float* begin() { return mat.data(); }
        float* end() { return mat.data() + mat.size(); }
        
        const float* begin() const { return mat.data(); }
        const float* end() const { return mat.data() + mat.size(); }
        
        float& operator[](size_t idx) { return mat.data()[idx]; }
        float operator[](size_t idx) const { return mat.data()[idx]; }
        
        float& at(size_t idx) { return mat.data()[idx]; }
        float at(size_t idx) const { return mat.data()[idx]; }
        
        size_t size() const { return mat.size(); }
    };

    /// @brief Column vector wrapper around Eigen::VectorXf
    struct Vector {
        int dim;
        Matrix data;  // Stored as Nx1 matrix for compatibility
        
        Vector(std::vector<float>&& source) 
            : dim(static_cast<int>(source.size())), 
              data(std::move(source), static_cast<int>(source.size()), 1) {
        }
        
        explicit Vector(Matrix&& vector_as_matrix) 
            : dim(vector_as_matrix.R),
              data(std::move(vector_as_matrix)) {
        }
        
        Vector(int dim) : dim(dim), data(dim, 1) {}
        
        // Default constructor
        Vector() : dim(1), data(1, 1) {}
        
        // Default move and copy semantics
        Vector(const Vector&) = default;
        Vector(Vector&&) = default;
        Vector& operator=(const Vector&) = default;
        Vector& operator=(Vector&&) = default;
        
        inline float& atr(int i) {
            return data.atr(i, 0);
        }
        
        inline float atc(int i) const {
            return data.atc(i, 0);
        }
    };

    // Matrix multiplication A*B (using Eigen for SIMD optimization)
    inline Matrix mul(const Matrix& A, const Matrix& B) {
        assert(A.C == B.R && "Mismatching matrix dimensions for multiplication!");
        Matrix M{A.R, B.C};
        M.mat = A.mat * B.mat;
        return M;
    }

    // A * B^T (transposed multiplication, lazy evaluation)
    inline Matrix mult(const Matrix& A, const Matrix& B) {
        assert(A.C == B.C && "Mismatching matrix dimensions for multiplication!");
        Matrix M{A.R, B.R};
        M.mat = A.mat * B.mat.transpose();
        return M;
    }

    // A^T * B (transposed multiplication, lazy evaluation with no copy)
    inline Matrix tmul(const Matrix& A, const Matrix& B) {
        assert(A.R == B.R && "Mismatching matrix dimensions for multiplication!");
        Matrix M{A.C, B.C};
        M.mat = A.mat.transpose() * B.mat;
        return M;
    }

    // Element-wise addition A + B
    inline Matrix add(const Matrix& A, const Matrix& B) {
        assert(A.R == B.R && A.C == B.C && "Mismatching matrix dimensions for addition!");
        Matrix M{A.R, A.C};
        M.mat = A.mat + B.mat;
        return M;
    }

    // Element-wise multiplication (Hadamard product)
    inline Matrix muld(const Matrix& A, const Matrix& B) {
        assert(A.R == B.R && A.C == B.C && "Mismatching matrix dimensions for direct product!");
        Matrix M{A.R, A.C};
        M.mat = A.mat.cwiseProduct(B.mat);
        return M;
    }

    // Element-wise subtraction A - B
    inline Matrix sub(const Matrix& A, const Matrix& B) {
        assert(A.R == B.R && A.C == B.C && "Mismatching matrix dimensions for subtraction!");
        Matrix M{A.R, A.C};
        M.mat = A.mat - B.mat;
        return M;
    }

    // Scalar multiplication
    inline Matrix scale(float scale, const Matrix& A) {
        Matrix M{A.R, A.C};
        M.mat = scale * A.mat;
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
    
    // In-place operations using Eigen's .noalias() for efficiency (avoids temporary matrices)
    inline Matrix& operator+=(Matrix& A, const Matrix& B) { 
        A.mat.noalias() += B.mat;
        return A;
    }
    inline Matrix& operator-=(Matrix& A, const Matrix& B) { 
        A.mat.noalias() -= B.mat;
        return A;
    }

    // Matrix-vector multiplication A*v
    inline Vector mul(const Matrix& A, const Vector& v) {
        assert(A.C == v.dim && "Mismatching matrix-vector dimensions for multiplication!");
        Vector R{A.R};
        R.data.mat = A.mat * v.data.mat;
        return R;
    }

    // Outer product of two vectors a*b^T (using Eigen expression templates)
    inline Matrix mulo(const Vector& a, const Vector& b) {
        Matrix M{a.dim, b.dim};
        M.mat = a.data.mat * b.data.mat.transpose();
        return M;
    }

    // Vector-vector operations
    inline Vector add(const Vector& a, const Vector& b) {
        Vector R{a.dim};
        R.data.mat = a.data.mat + b.data.mat;
        return R;
    }
    
    inline Vector sub(const Vector& a, const Vector& b) {
        Vector R{a.dim};
        R.data.mat = a.data.mat - b.data.mat;
        return R;
    }

    inline Vector muld(const Vector& a, const Vector& b) {
        Vector R{a.dim};
        R.data.mat = a.data.mat.cwiseProduct(b.data.mat);
        return R;
    }
    
    inline Vector scale(float scaler, const Vector& v) {
        Vector R{v.dim};
        R.data.mat = scaler * v.data.mat;
        return R;
    }

    inline Vector operator+(const Vector& a, const Vector& b) { return add(a, b); }
    inline Vector operator-(const Vector& a, const Vector& b) { return sub(a, b); }
    inline Vector operator*(const Matrix& M, const Vector& v) { return mul(M, v); }
    inline Vector operator*(float scaler, const Vector& v) { return scale(scaler, v); }
    
    // In-place operations using Eigen's .noalias() (avoids temporary vectors)
    inline Vector& operator+=(Vector& a, const Vector& b) {
        a.data.mat.noalias() += b.data.mat;
        return a;
    }
    inline Vector& operator-=(Vector& a, const Vector& b) {
        a.data.mat.noalias() -= b.data.mat;
        return a;
    }
}