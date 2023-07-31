#include <iostream>
#include "hls_stream.h"
#include "ap_int.h"

#define WIDTH 10
#define HEIGHT 10
#define INT_WIDTH	8

void conv2d(hls::stream<int>& out_stream, hls::stream<int>& in_stream);

// Test image
int test_image[HEIGHT][WIDTH] = {
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    {11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    {21, 22, 23, 24, 25, 26, 27, 28, 29, 30},
    {31, 32, 33, 34, 35, 36, 37, 38, 39, 40},
    {41, 42, 43, 44, 45, 46, 47, 48, 49, 50},
    {51, 52, 53, 54, 55, 56, 57, 58, 59, 60},
    {61, 62, 63, 64, 65, 66, 67, 68, 69, 70},
    {71, 72, 73, 74, 75, 76, 77, 78, 79, 80},
    {81, 82, 83, 84, 85, 86, 87, 88, 89, 90},
    {91, 92, 93, 94, 95, 96, 97, 98, 99, 100}
};

int main()
{
	int result;

    hls::stream<int> in_stream("in_stream_test");
    hls::stream<int> out_stream("out_stream_test");

    // Write test image to input stream
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {

                in_stream<<test_image[y][x];
            }
        }

        // Call the conv2d function
    conv2d(out_stream, in_stream);

    // Read output from out_stream and print the result
    std::cout << "Filtered Output:" << std::endl;
    for (int y = 0; y < HEIGHT; y++) {

        for (int x = 0; x < WIDTH; x++) {
        		out_stream>>result;
        	    std::cout <<result << "\t";
        	}
        std::cout << std::endl;

        }

    return 0;
}
