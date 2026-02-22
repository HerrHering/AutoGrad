#include <iostream>
#include <MUtils.hpp>

using namespace MUtils;

int main() {
    std::cout << "Hello World!\n";

#ifndef NDEBUG
    std::cout << "Debug mode enabled." << std::endl;
#else
    std::cout << "Release mode enabled." << std::endl;
#endif

    Matrix m{std::vector<float>{1, 2, 3, 4}, 2, 2};

    std::cout << to_string(tmul(m, m)) << std::endl;
    std::cout << to_string(mult(m, m)) << std::endl;
}