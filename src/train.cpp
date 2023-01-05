#include "train.h"
#include "dataset.h"

#include <chrono>
#include <iostream>

DataEntry entries[BATCH_SIZE];

void train(const std::string &networkName, const std::string &trainPath, const std::string &validationPath, Model &model) {

    Dataset trainingData = Dataset(trainPath);

    std::cout << "\nStarted training " << networkName << "!" << std::endl;

    unsigned int iteration = 0;
    float totalError = 0;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (unsigned int epoch = 1; epoch < EPOCHS; epoch++) {

        bool newEpoch = false;
        while (!newEpoch) {
            iteration++;
            newEpoch = trainingData.readEntries(entries);

            float e = 0;
            float L_1_BIAS_GRADIENT = 0;
            float L_1_WEIGHT_GRADIENT[L_1_SIZE] = {0};

            float L_0_BIAS_GRADIENT[L_1_SIZE] = {0};
            float L_0_WEIGHT_GRADIENT[L_0_SIZE * L_1_SIZE] = {0};

            for (unsigned int batchIdx = 0; batchIdx < BATCH_SIZE; batchIdx++) {

                DataEntry &entry = entries[batchIdx];

                float hiddenLayer[L_1_SIZE];

                float output = sigmoid(model.forward(entry, hiddenLayer));
                e += error(output, entry.expected);

                float lossOutput = sigmoidDerivative(output) * errorDerivative(output, entry.expected);
                float lossHiddenLayer[L_1_SIZE];

                for (int idx = 0; idx < L_1_SIZE; idx++) {
                    lossHiddenLayer[idx] = lossOutput * model.L_1.getWeight(idx, 0) * ReLUDerivative(hiddenLayer[idx]);
                }

                L_1_BIAS_GRADIENT += lossOutput;
                for (int i = 0; i < L_1_SIZE * L_2_SIZE; i++) {
                    L_1_WEIGHT_GRADIENT[i] += hiddenLayer[i] * lossOutput;
                }

                for (int i = 0; i < L_1_SIZE; i++) {
                    L_0_BIAS_GRADIENT[i] += lossHiddenLayer[i];
                }

                for (unsigned int idx : entry.whiteFeatureIndexes) {
                    for (int i = 0; i < L_1_SIZE; i++) {
                        L_0_WEIGHT_GRADIENT[idx * L_1_SIZE + i] += lossHiddenLayer[i];
                    }
                }
            }

            model.L_1.biases[0] -= L_1_BIAS_GRADIENT * LR;
            for (int i = 0; i < L_1_SIZE * L_2_SIZE; i++) {
                model.L_1.weights[i] -= L_1_WEIGHT_GRADIENT[i] * LR;
            }
            for (int i = 0; i < L_1_SIZE; i++) {
                model.L_0.biases[i] -= L_0_BIAS_GRADIENT[i] * LR;
            }
            for (int i = 0; i < L_0_SIZE * L_1_SIZE; i++) {
                model.L_0.weights[i] -= L_0_WEIGHT_GRADIENT[i] * LR;
            }

            totalError += e;

            if (iteration % ITERATIONS_PER_CHECKPOINT == 0) {
                float averageError = totalError / ITERATIONS_PER_CHECKPOINT / BATCH_SIZE;
                std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                long secondsSinceEpoch = std::chrono::duration_cast<std::chrono::seconds>(now - begin).count();
                long positionsPerSecond = iteration * BATCH_SIZE / secondsSinceEpoch;
                std::cout << "Epoch " << epoch << " - Iteration " << iteration << " - Error " << averageError << " - Elapsed time " << secondsSinceEpoch << "s - Position/second " << positionsPerSecond << "\r" << std::flush;
                totalError = 0;
            }
            // Forward DONE
            // Loss DONE
            // Backwards gradient DONE
        }

        system("mkdir nets");
        std::cout << "\nEpoch " << epoch << " has finished!" << std::endl;
        FILE *f;
        f = fopen(("nets/" + networkName + "_" + std::to_string(epoch) + ".bin").c_str(), "wb");
        model.writeToFile(f);
        fclose(f);
    }

    trainingData.close();
}