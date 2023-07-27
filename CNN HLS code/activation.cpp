#include <cmath>

float swish(float x) {
    return x / (1.0 + exp(-x));
}
