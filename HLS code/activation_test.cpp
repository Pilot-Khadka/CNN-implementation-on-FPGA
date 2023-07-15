#include<iostream>

float swish(float x);

int main(){
	float input = 2.5;
	float result = swish(input);
	std::cout<<"swish:" << result << std::endl;
	return 0;
}
