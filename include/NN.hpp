#pragma once

#include <MUtils.hpp>

namespace NN {
    enum class ActivationFunctionTypes { Tanh };

    template <ActivationFunctionTypes Type>
    struct Layer {
        int dim;
        
    };
}