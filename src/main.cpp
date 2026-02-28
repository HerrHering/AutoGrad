#include <iostream>
#include <MUtils.hpp>
#include <NN.hpp>

using namespace MUtils;
using namespace NN;

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

    // input(5) -> layer1(10) -> layer2(2)
    Layer<ActivationType::ReLU> layer1{NNUtils::initWeigths(10, 5, ActivationType::ReLU), Vector(10)};
    Layer<ActivationType::ReLU> layer2{NNUtils::initWeigths(2, 10, ActivationType::Sigmoid), Vector(2)};
    Network network(std::move(layer1), std::move(layer2));
}