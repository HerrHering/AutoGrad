#include <iostream>
#include <MUtils.hpp>

using namespace MUtils;

int main() {
    std::cout << "Hello World!\n";

    Matrix m{std::vector<float>{1, 2, 3, 4}, 2, 2};

    std::cout << to_string(tmul(m, m)) << std::endl;
}