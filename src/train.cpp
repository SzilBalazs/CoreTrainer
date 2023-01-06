#include "train.h"
#include "adam.h"
#include "dataset.h"
#include "gradient.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

float LR = 0.001;

DataEntry entries[BATCH_SIZE];
std::vector<Gradient> gradients(THREAD_COUNT);
std::vector<std::thread> ths(THREAD_COUNT);
float errors[THREAD_COUNT];

void processBatch(Model &model, int threadId) {

    Gradient &grad = gradients[threadId];

    grad.reset();
    alignas(64) float hiddenLayer[L_1_SIZE];
    alignas(64) float hiddenLayerLoss[L_1_SIZE];
    errors[threadId] = 0;

    for (unsigned int batchIdx = threadId; batchIdx < BATCH_SIZE; batchIdx += THREAD_COUNT) {

        DataEntry &entry = entries[batchIdx];

        float output = sigmoid(model.forward(entry, hiddenLayer));
        errors[threadId] += error(output, entry.expected);

        float outputLoss = sigmoidDerivative(output) * errorDerivative(output, entry.expected);

        for (unsigned int idx = 0; idx < L_1_SIZE; idx++) {
            hiddenLayerLoss[idx] = outputLoss * model.L_1.getWeight(idx, 0) * ReLUDerivative(hiddenLayer[idx]);
        }

        grad.L_1_BIAS_GRADIENT += outputLoss;
        for (unsigned int i = 0; i < L_1_SIZE * L_2_SIZE; i++) {
            grad.L_1_WEIGHT_GRADIENT[i] += hiddenLayer[i] * outputLoss;
        }

        for (unsigned int i = 0; i < L_1_SIZE; i++) {
            grad.L_0_BIAS_GRADIENT[i] += hiddenLayerLoss[i];
        }

        for (unsigned int idx : entry.whiteFeatureIndexes) {
            for (unsigned int i = 0; i < L_1_SIZE; i++) {
                grad.L_0_WEIGHT_GRADIENT[idx * L_1_SIZE + i] += hiddenLayerLoss[i];
            }
        }

        /*for (unsigned int idx : entry.blackFeatureIndexes) {
            for (unsigned int i = 0; i < L_1_SIZE; i++) {
                grad.L_0_WEIGHT_GRADIENT[idx * L_1_SIZE + i] += hiddenLayerLoss[i];
            }
        }*/
    }
}

void train(const std::string &networkName, const std::string &trainPath, const std::string &validationPath, Model &model) {

    Dataset trainingData = Dataset(trainPath);

    if (!std::filesystem::exists("nets/")) {
        std::filesystem::create_directory("nets/");
    }

    if (std::filesystem::exists("loss.txt")) {
        std::filesystem::remove("loss.txt");
    }

    std::cout << "\nStarted training " << networkName << "!" << std::endl;

    Adam adam = Adam();

    unsigned int iteration = 0;
    float totalError = 0;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (unsigned int epoch = 1; epoch < EPOCHS; epoch++) {

        bool newEpoch = false;
        while (!newEpoch) {
            iteration++;
            newEpoch = trainingData.readEntries(entries);

            for (int id = 0; id < THREAD_COUNT; id++) {
                ths[id] = std::thread(processBatch, std::ref(model), id);
            }

            for (int id = 0; id < THREAD_COUNT; id++) {
                if (ths[id].joinable())
                    ths[id].join();
            }

            adam.applyGradients(gradients, model);

            for (int id = 0; id < THREAD_COUNT; id++) {
                totalError += errors[id];
            }

            if (iteration % ITERATIONS_PER_CHECKPOINT == 0) {
                float averageError = totalError / ITERATIONS_PER_CHECKPOINT / BATCH_SIZE;
                std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                long secondsSinceEpoch = std::chrono::duration_cast<std::chrono::seconds>(now - begin).count();
                long positionsPerSecond = iteration * BATCH_SIZE / (secondsSinceEpoch + 1);
                std::cout << "                                               \r" << std::flush;
                std::cout << "Epoch " << epoch << " - Iteration " << iteration << " - Error " << averageError << " - Elapsed time " << secondsSinceEpoch << "s - Position/second " << positionsPerSecond << "\r" << std::flush;
                totalError = 0;

                std::ofstream data("loss.txt", std::ios::app);
                data << iteration << " " << averageError << "\n";
                data.close();
                system("python3 graph.py &");
            }
        }

        if (epoch == 30) {
            LR /= 10;
        }
        std::cout << "\nEpoch " << epoch << " has finished!" << std::endl;
        FILE *f;
        f = fopen(("nets/" + networkName + "_" + std::to_string(epoch) + ".bin").c_str(), "wb");
        model.writeToFile(f);
        fclose(f);
    }

    trainingData.close();
}