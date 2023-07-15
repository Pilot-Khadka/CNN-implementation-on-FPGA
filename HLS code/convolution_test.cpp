#include<iostream>
#include <cstdlib>

void convolution(int(*matrix)[5], int rows, int cols, int(*kernel)[3],int krows, int kcols,int (*result)[3]);

int main() {
	const int matrix_rows = 5;
	const int matrix_cols = 5;
	const int kernel_rows = 3;
    const int kernel_cols = 3;

    int matrix[matrix_rows][matrix_cols];
    int kernel[kernel_rows][kernel_cols] = {
    	{-1, 0, 1},
	    {-1, 0, 2},
	    {-1, 0, 1}
	};

	std::srand(100); // seed value
    for (int i = 0; i < matrix_rows; ++i) {
    	for (int j = 0; j < matrix_cols; ++j) {
    		matrix[i][j] = std::rand() % 10; //generate random values between 0 and 9
	        }
	    }
    for (int i = 0; i < matrix_rows; ++i) {
    	for (int j = 0; j < matrix_cols; ++j) {
    		std::cout << matrix[i][j] << " ";
    	}
    	std::cout << std::endl;
    }

    const int result_rows = matrix_rows - kernel_rows + 1;
    const int result_cols = matrix_cols - kernel_cols + 1;
    int result[result_rows][result_cols] = {0};

    convolution(matrix,matrix_rows,matrix_cols,kernel,kernel_rows,kernel_cols,result);

    for (int i = 0; i < result_rows; ++i) {
        	for (int j = 0; j < result_cols; ++j) {
        		std::cout << result[i][j] << " ";
        	}
        	std::cout << std::endl;
        }

    return 0;
}
