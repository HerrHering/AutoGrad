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
struct WdbcRecord {
    Vector features{30};
    Vector label{2}; // 0 or 1
};
// Simple parser logic
std::vector<WdbcRecord> loadWdcb(std::string filename) {
    std::vector<WdbcRecord> data;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        WdbcRecord record;
        std::getline(ss, val, ','); // ID
        std::getline(ss, val, ','); // Label
        if (val == "M") record.label.atr(0) = 1.0f; // M
        else record.label.atr(1) = 1.0f; // B
        // Features
        for (int i = 0; i < 30; ++i) {
            std::getline(ss, val, ',');
            record.features.atr(i) = std::stof(val);
        }
        data.push_back(record);
    }
    return data;
}
void model_scores(const std::vector<WdbcRecord>& test_data, auto network) {
    const int table_size = 2;
    // Row: True
    // Column: Predicted
    int confusion_matrix[table_size][table_size] = {0};
    for (const auto& sample : test_data) {
        auto prediction = network.forward(sample.features);
        
        // Get the index of the highest value (Argmax)
        int predicted_label = std::distance(prediction.data.data.begin(), 
                                            std::max_element(prediction.data.data.begin(), prediction.data.data.end()));
        int true_label = std::distance(sample.label.data.data.begin(), 
                                            std::max_element(sample.label.data.data.begin(), sample.label.data.data.end()));
        
        confusion_matrix[true_label][predicted_label]++;
    }
    // Calculate Metrics
    int total_correct = 0;
    for(int i=0; i<table_size; ++i) total_correct += confusion_matrix[i][i];

    std::cout << "\n--- Evaluation ---\n";
    std::cout << RED << "Overall Accuracy: " << (double)total_correct / test_data.size() * 100.0 << "%\n\n" << RESET;

    for (int i = 0; i < table_size; ++i) {
        std::string name = (i == 0) ? std::string(MAGENTA).append("bening   ") : std::string(BLUE).append("malignant");
        
        double TP = confusion_matrix[i][i];
        double FN = 0, FP = 0, TN = 0;

        for (int j = 0; j < table_size; ++j) {
            if (i != j) {
                FN += confusion_matrix[i][j]; // Actual i, but predicted j
                FP += confusion_matrix[j][i]; // Actual j, but predicted i
            }
        }
        
        // TN is everything else
        TN = test_data.size() - (TP + FN + FP);

        double TPR = (TP + FN > 0) ? (TP / (TP + FN)) : 0; // Sensitivity
        double TNR = (TN + FP > 0) ? (TN / (TN + FP)) : 0; // Specificity

        std::cout << name << " | TPR: " << TPR << " | TNR: " << TNR << "\n";
    }
}
void minMaxScale(std::vector<WdbcRecord>& data) {
    if (data.empty()) return;

    int num_features = 30;
    std::vector<float> min_vals(num_features, std::numeric_limits<float>::max());
    std::vector<float> max_vals(num_features, std::numeric_limits<float>::lowest());

    // 1. Pass through the data to find Min and Max for each feature
    for (const auto& record : data) {
        for (int i = 0; i < num_features; ++i) {
            float val = record.features.atc(i); // Using your atc() const accessor
            if (val < min_vals[i]) min_vals[i] = val;
            if (val > max_vals[i]) max_vals[i] = val;
        }
    }

    // 2. Pass through the data again to scale every value
    for (auto& record : data) {
        for (int i = 0; i < num_features; ++i) {
            float range = max_vals[i] - min_vals[i];
            
            // Handle edge case: if min == max, range is 0. Set feature to 0.
            if (range > 0.0f) {
                float original = record.features.atc(i);
                record.features.atr(i) = (original - min_vals[i]) / range;
            } else {
                record.features.atr(i) = 0.0f; 
            }
        }
    }
}

int main() {
#ifndef NDEBUG
    std::cout << "Debug mode enabled." << std::endl;
#else
    std::cout << "Release mode enabled." << std::endl;
#endif

    Network network{
        make_layer<ActivationType::ReLU>(30, 16),
        make_layer<ActivationType::ReLU>(16, 16),
        make_layer<ActivationType::Sigmoid>(16, 2)
    };

    auto wdbc_data = loadWdcb("assets/wdbc.data");
    minMaxScale(wdbc_data);

    // Split data
    std::random_device rd{};
    std::mt19937 g(rd());
    // I dont trust shuffling lowkey
    std::shuffle(wdbc_data.begin(), wdbc_data.end(), g);
    std::shuffle(wdbc_data.begin(), wdbc_data.end(), g);
    std::shuffle(wdbc_data.begin(), wdbc_data.end(), g);

    // 80% train
    int train_size = (int)((double)wdbc_data.size() * 0.8);
    std::cout << YELLOW << "Data set size: " << wdbc_data.size() << "\n" << RESET;
    std::vector<WdbcRecord> train_data(wdbc_data.begin(), wdbc_data.begin() + train_size);
    std::cout << YELLOW << "Train set size: " << train_data.size() << "\n" << RESET;
    std::vector<WdbcRecord> test_data(wdbc_data.begin() + train_size, wdbc_data.end());

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
    model_scores(test_data, network);
}