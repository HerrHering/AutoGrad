#pragma once

#include <MUtils.hpp>
#include <random>

namespace NN {
    using MUtils::Vector, MUtils::Matrix;

    enum class ActivationType {ReLU, Sigmoid};
    enum class LossType { MSE };
    namespace NNUtils {
        Matrix initWeigths(int r, int c, ActivationType type) {
            Matrix weights{r, c};

            // Initialize for sigmoid / tanh
            static std::mt19937 gen(std::random_device{}());

            // Xavier Initialization: sqrt(2 / (fan_in + fan_out))
            float stddev;
            if (type == ActivationType::ReLU) {
                stddev = std::sqrt(2.0f / (float)c);
            }
            else {
                stddev = std::sqrt(2.0f / (float)(r + c));
            }
            
            std::normal_distribution<float> dist(0.0f, stddev);

            for (auto& w : weights.data) {
                w = dist(gen);
            }

            return weights;
        }
    };

    struct layer_backward_res {
        Vector dL_da;
        Matrix d_weights;
        Vector d_biases;
    };
    template <ActivationType Type>
    struct Layer {
        Layer(Matrix&& weights, Vector&& biases)
            : weights(std::move(weights)), biases(std::move(biases)),
            last_activation(1), last_input(1), last_z(1) {}
        // Weights outgoing from PREVIOUS layer to the THIS layer
        Matrix weights;
        Vector biases;
        
        // a = activation(Z)
        Vector last_activation;
        Vector last_input;
        // Z = Wx + b
        Vector last_z;
        Vector forward(const Vector& input) {
            last_input = input;
            last_z = weights * input + biases;
            last_activation = last_z;

            // Apply formula
            for (int i = 0; i < last_activation.dim; i++) {
                if constexpr (Type == ActivationType::ReLU) {
                    last_activation.atr(i) = std::max(0.0f, last_activation.atc(i));
                } else if constexpr (Type == ActivationType::Sigmoid) {
                    last_activation.atr(i) = 1.0f / (1.0f + std::exp(-last_activation.atc(i)));
                }
            }

            return last_activation;
        }

        /// @brief In: dL/da_i, where this is the i-th layer
        /// @return Out: dL/da_{i-1}, so the loss propagated to the previous layer
        layer_backward_res backward(const Vector& dL_da) {
            // a = activate(z)
            // z = W*a_prev + b
            // delta = dL/dz = dL/da * da/dz = dL/da * activation'(z)
            Vector delta = dL_da;

            // Apply derivative
            for (int i = 0; i < delta.dim; i++) {
                if constexpr (Type == ActivationType::ReLU) {
                    // d(ReLU)/dz = 1 if z > 0, else 0
                    delta.atr(i) *= last_z.atc(i) > 0 ? 1.0f : 0.0f;
                } else if constexpr (Type == ActivationType::Sigmoid) {
                    delta.atr(i) *= last_activation.atc(i) * (1.0f - last_activation.atc(i));
                }
            }

            // dL/da' = (dL/da * da/dz) * dz/da' <- W = W^T * delta
            #pragma message("tmul() is not implemented for vector arguments yet. This might be slower.")
            return layer_backward_res{
                .dL_da = Vector(MUtils::tmul(weights, delta.data)),
                .d_weights = MUtils::mulo(delta, last_input),
                .d_biases = delta
            };
        }
    };
    template <ActivationType Type>
    Layer<Type> make_layer(int input, int output) {
        return Layer<Type>{
            NNUtils::initWeigths(output, input, Type),
            Vector(output)
        };
    }

    template <typename... Layers>
    struct Network {
        std::tuple<Layers...> layers;
        Network(Layers...  layers) : layers(std::move(layers)...), last_pred(1) {}

        Vector last_pred;
        Vector forward(const Vector& input) {
            // Copy
            Vector temp = input;
            std::apply([&](auto&... layer) {
                ((temp = layer.forward(temp)), ...);
            }, layers);

            last_pred = temp;
            return temp;
        }
        template <LossType Loss>
        void backward(const Vector& target, float learning_rate) {
            Vector loss_grad{last_pred.dim};
            // Calculate loss gradient wrt the last activation layer
            if constexpr (Loss == LossType::MSE) {
                for (int i = 0; i < loss_grad.dim; i++) {
                    // L = (target - pred)^2 -> dL/d(pred) = 2(target - pred)*-1 = 2*(pred - target)
                    loss_grad.atr(i) = last_pred.atc(i) - target.atc(i);
                }
            }

            // Iterate through the layers in reverse order
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                ((
                    [&] {
                        auto& layer = std::get<sizeof...(Layers) - 1 - Is>(layers);
                        layer_backward_res grads = layer.backward(loss_grad);

                        // Update parameters
                        layer.weights = layer.weights - learning_rate * grads.d_weights;
                        layer.biases = layer.biases - learning_rate * grads.d_biases;
                        // Propagate error
                        loss_grad = std::move(grads.dL_da);
                    }()
                ), ...);
            } (std::make_index_sequence<sizeof...(Layers)>{});
        }
    };
}