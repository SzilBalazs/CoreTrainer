#ifndef CORETRAINER_SRC_ADAM_H_
#define CORETRAINER_SRC_ADAM_H_

#include "gradient.h"
#include "model.h"

#include <cmath>
#include <vector>

struct Adam {
    Gradient mGradient, vGradient;

    inline Adam() {
        mGradient.reset();
        vGradient.reset();
    }

    inline void applyGradient(float &target, float &m, float &v, float gradient) {
        m = BETA1 * m + (1.0 - BETA1) * gradient;
        v = BETA2 * v + (1.0 - BETA2) * gradient * gradient;

        target -= LR * m / (std::sqrt(v) + EPSILON);
    }

    inline void applyGradients(const std::vector<Gradient> &gradients, Model &model) {

        float totalGrad = 0;
        for (const Gradient &grad : gradients) {
            totalGrad += grad.L_1_BIAS_GRADIENT;
        }

        applyGradient(model.L_1.biases[0], mGradient.L_1_BIAS_GRADIENT, vGradient.L_1_BIAS_GRADIENT, totalGrad);

        for (unsigned int i = 0; i < L_1_SIZE * L_2_SIZE; i++) {
            totalGrad = 0;
            for (const Gradient &grad : gradients) {
                totalGrad += grad.L_1_WEIGHT_GRADIENT[i];
            }
            applyGradient(model.L_1.weights[i], mGradient.L_1_WEIGHT_GRADIENT[i], vGradient.L_1_WEIGHT_GRADIENT[i], totalGrad);
        }

        for (unsigned int i = 0; i < L_1_SIZE; i++) {
            totalGrad = 0;
            for (const Gradient &grad : gradients) {
                totalGrad += grad.L_0_BIAS_GRADIENT[i];
            }
            applyGradient(model.L_0.biases[i], mGradient.L_0_BIAS_GRADIENT[i], vGradient.L_0_BIAS_GRADIENT[i], totalGrad);
        }

        for (unsigned int i = 0; i < L_0_SIZE * L_1_SIZE; i++) {
            totalGrad = 0;
            for (const Gradient &grad : gradients) {
                totalGrad += grad.L_0_WEIGHT_GRADIENT[i];
            }
            applyGradient(model.L_0.weights[i], mGradient.L_0_WEIGHT_GRADIENT[i], vGradient.L_0_WEIGHT_GRADIENT[i], totalGrad);
        }
    }
};

#endif //CORETRAINER_SRC_ADAM_H_
