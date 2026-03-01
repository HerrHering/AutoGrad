#include <MUtils.hpp>
#include <NN.hpp>

using namespace MUtils;
using namespace NN;

#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>
#include <colors.hpp>

// Data loading
struct IrisRecord {
    Vector features{4};
    Vector label{3}; // 0, 1, or 2
};
// Simple parser logic
std::vector<IrisRecord> loadIris(std::string filename) {
    std::vector<IrisRecord> data;
    std::ifstream file(filename);
    std::string line;
    
    std::getline(file, line); // Skip header
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        IrisRecord record;
        for (int i = 0; i < 4; ++i) {
            std::getline(ss, val, ',');
            record.features.atr(i) = std::stof(val);
        }
        std::getline(ss, val, ',');
        if (val == "\"Setosa\"") record.label.atr(0) = 1.0f;
        else if (val == "\"Versicolor\"") record.label.atr(1) = 1.0f;
        else record.label.atr(2) = 1.0f;
        data.push_back(record);
    }
    return data;
}

int main() {
#ifndef NDEBUG
    std::cout << "Debug mode enabled." << std::endl;
#else
    std::cout << "Release mode enabled." << std::endl;
#endif

    // input(4) -> layer1(5) -> layer2(5) -> layer3(3)
    Network network{
        make_layer<ActivationType::ReLU>(4, 15),
        make_layer<ActivationType::Sigmoid>(15, 3),
    };

    auto flower_data = loadIris("assets/iris.csv");

    // Split data
    std::random_device rd{};
    std::mt19937 g(rd());
    std::shuffle(flower_data.begin(), flower_data.end(), g);

    // 80% train
    int train_size = (int)((double)flower_data.size() * 0.8);
    std::cout << YELLOW << "Data set size: " << flower_data.size() << "\n" << RESET;
    std::vector<IrisRecord> train_data(flower_data.begin(), flower_data.begin() + train_size);
    std::cout << YELLOW << "Train set size: " << train_data.size() << "\n" << RESET;
    std::vector<IrisRecord> test_data(flower_data.begin() + train_size, flower_data.end());

    // Training loop
    for (int epoch = 0; epoch < 1000; epoch++) {
        double total_loss = 0;
        // Important: Shuffle data to help generalization
        std::shuffle(train_data.begin(), train_data.end(), g);

        // Shuffle data to avoid distribution bias
        for (const auto& sample : train_data) {
            // Forward
            auto prediction = network.forward(sample.features);

            // Loss
            for (int i = 0; i < prediction.dim; i++) {
                float error = sample.label.atc(i) - prediction.atc(i);
                total_loss += error * error;
            }

            // Backward
            network.backward<LossType::MSE>(sample.label, 0.01f);
        }

        // Print progress every 100 epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch 
                    << " | Loss: " << (total_loss / train_data.size()) 
                    << std::endl;
        }
    }

    // Print predictions
    for (const auto& sample : test_data) {
        auto prediction = network.forward(sample.features);
        
        // Get the index of the highest value (Argmax)
        int predicted_label = std::distance(prediction.data.data.begin(), 
                                            std::max_element(prediction.data.data.begin(), prediction.data.data.end()));
        int true_label = std::distance(sample.label.data.data.begin(), 
                                            std::max_element(sample.label.data.data.begin(), sample.label.data.data.end()));
        
        std::cout << (true_label == predicted_label ? GREEN : RED)  << "True label: " << true_label << ", Predicted: " << predicted_label << "\n" << RESET;
    }
}