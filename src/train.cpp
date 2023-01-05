#include "train.h"
#include "dataset.h"

#include <iostream>

DataEntry entries[BATCH_SIZE];

void train(const std::string &networkName, const std::string &trainPath, const std::string &validationPath, Model &model) {

    Dataset trainingData = Dataset(trainPath);

    std::cout << "\nStarted training " << networkName << "!" << std::endl;

    unsigned int iteration = 0;
    float totalError = 0;
    for (unsigned int epoch = 1; epoch < EPOCHS; epoch++) {

        bool newEpoch = false;
        while (!newEpoch) {
            iteration++;
            newEpoch = trainingData.readEntries(entries);

            float e = 0;
            float L_1_BIAS_GRADIENT = 0;
            float L_1_WEIGHT_GRADIENT[2 * L_1_SIZE * L_2_SIZE] = {0};

            float L_0_BIAS_GRADIENT[L_1_SIZE] = {0};
            float L_0_WEIGHT_GRADIENT[L_0_SIZE * L_1_SIZE] = {0};

            for (unsigned int batchIdx = 0; batchIdx < BATCH_SIZE; batchIdx++) {

                DataEntry &entry = entries[batchIdx];

                float hiddenLayer[L_1_SIZE * 2];

                float output = model.forward(entry, hiddenLayer);
                e += error(output, entry.wdl, entry.eval);

                float lossOutput = sigmoidDerivative(output) * errorDerivative(output, entry.wdl, entry.eval);
                float lossHiddenLayer[L_1_SIZE * 2];

                for (int idx = 0; idx < L_1_SIZE * 2; idx++) {
                    lossHiddenLayer[idx] = lossOutput * model.L_1.getWeight(idx, 0) * clippedReLUDerivative(hiddenLayer[idx]);
                }

                L_1_BIAS_GRADIENT += lossOutput;
                for (int i = 0; i < 2 * L_1_SIZE * L_2_SIZE; i++) {
                    L_1_WEIGHT_GRADIENT[i] += hiddenLayer[i] * lossOutput;
                }

                for (int i = 0; i < L_1_SIZE; i++) {
                    L_0_BIAS_GRADIENT[i] += lossHiddenLayer[i] + lossHiddenLayer[i + L_1_SIZE];
                }

                for (unsigned int idx : entry.whiteFeatureIndexes) {
                    for (int i = 0; i < L_1_SIZE; i++) {
                        L_0_WEIGHT_GRADIENT[idx * L_1_SIZE + i] += lossHiddenLayer[i];
                    }
                }

                for (unsigned int idx : entry.blackFeatureIndexes) {
                    for (int i = 0; i < L_1_SIZE; i++) {
                        L_0_WEIGHT_GRADIENT[idx * L_1_SIZE + i] += lossHiddenLayer[i + L_1_SIZE];
                    }
                }
            }

            model.L_1.biases[0] -= L_1_BIAS_GRADIENT * LR;
            for (int i = 0; i < 2 * L_1_SIZE * L_2_SIZE; i++) {
                model.L_1.weights[i] -= L_1_WEIGHT_GRADIENT[i] * LR;
            }
            for (int i = 0; i < L_1_SIZE; i++) {
                model.L_0.biases[i] -= L_0_BIAS_GRADIENT[i] * LR;
            }
            for (int i = 0; i < L_0_SIZE * L_1_SIZE; i++) {
                model.L_0.weights[i] -= L_0_WEIGHT_GRADIENT[i] * LR;
            }

            totalError += e;

            if (iteration % 200 == 0) {
                std::cout << iteration << ": " << (totalError / 200 / BATCH_SIZE) << std::endl;
                totalError = 0;
            }
            // Forward DONE
            // Loss DONE
            // Backwards gradient TODO
        }
    }

    trainingData.close();
}