#include <iostream>

#include "PerceptronLayer.hpp"
#include "../0-Helpers/LossFunction.hpp"

class Sigmoid : public ActivationFunction {
public:
	double Activate(double input) {
		return 1.0 / (1.0 + exp(-input));
	}
	double Derivative(double input) {
		return Activate(input) * (1.0 - Activate(input));
	}

	std::vector<double> Activate(std::vector<double> input) {
		std::vector<double> output(input.size());
		for (int i = 0; i < input.size(); i++) {
			output[i] = Activate(input[i]);
		}
		return output;
	}
	std::vector<double> Derivative(std::vector<double> input) {
		std::vector<double> output(input.size());
		for (int i = 0; i < input.size(); i++) {
			output[i] = Derivative(input[i]);
		}
		return output;
	}
};

class MSE : public LossFunction {
public:
	double Calculate(double output, double target) {
		return pow(output - target, 2);
	}
	double Derivative(double output, double target) {
		return 2 * (output - target);
	}

	std::vector<double> Calculate(std::vector<double> output, std::vector<double> target) {
		std::vector<double> loss(output.size());
		for (int i = 0; i < output.size(); i++) {
			loss[i] = Calculate(output[i], target[i]);
		}
		return loss;
	}
	std::vector<double> Derivative(std::vector<double> output, std::vector<double> target) {
		std::vector<double> loss(output.size());
		for (int i = 0; i < output.size(); i++) {
			loss[i] = Derivative(output[i], target[i]);
		}
		return loss;
	}
};

int main() {
	PerceptronLayer p;
	p.Init(2,2, new Sigmoid());
	MSE mse;
	
	//solve the AND problem
	std::vector<std::vector<double>> x = { {0,0},{0,1},{1,0},{1,1} };
	std::vector<std::vector<double>> y = { {0,1},{0,1},{0,1},{1,0} };

	for (int i = 0; i < 1000000; i++) {
		for (int j = 0; j < x.size(); j++) {
			std::vector<double> o = p.Forward(x[j]);
			std::vector<double> dx = p.Backward(mse.Derivative(o, y[j]), 0.01);
		}
		printf("\033[0;0H");

		printf("0 AND 0 = %f,%f\tShould Be: 0,1\n", p.Forward({ 0,0 })[0],p.Forward({ 0,0 })[1]);
		printf("0 AND 1 = %f,%f\tShould Be: 0,1\n", p.Forward({ 0,1 })[0],p.Forward({ 0,1 })[1]);
		printf("1 AND 0 = %f,%f\tShould Be: 0,1\n", p.Forward({ 1,0 })[0],p.Forward({ 1,0 })[1]);
		printf("1 AND 1 = %f,%f\tShould Be: 1,0\n", p.Forward({ 1,1 })[0],p.Forward({ 1,1 })[1]);
	}

	return 0;
}