#include <stdio.h>
#define N 11
typedef int coef_t;
typedef int data_t;
typedef int acc_t;

// Function prototype
void fir(data_t *y, data_t x);

int main() {
    // Test input
    data_t x = 10;

    // Output variable
    data_t y;

    // Call the function
    fir(&y, x);

    // Print the result
    printf("Output: %d\n", y);

    return 0;
}
