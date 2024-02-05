#include "2DValidConvolutionLayer.hpp"
#include <random>
#include <time.h>
#include <thread>
#include <chrono>

#include "../0-Helpers/LossFunction.hpp"

class Sigmoid : public ActivationFunction
{
public:
	double Activate(double input)
	{
		return 1.0 / (1.0 + exp(-input));
	}
	double Derivative(double input)
	{
		double output = 1.0 / (1.0 + exp(-input)) * (1 - 1.0 / (1.0 + exp(-input)));
		return output;
	}

	std::vector<double> Activate(std::vector<double> input)
	{
		std::vector<double> output(input.size());
		for (int i = 0; i < input.size(); i++)
		{
			output[i] = Activate(input[i]);
		}
		return output;
	}
	std::vector<double> Derivative(std::vector<double> input)
	{
		std::vector<double> output(input.size());
		for (int i = 0; i < input.size(); i++)
		{
			output[i] = Derivative(input[i]);
		}
		return output;
	}
};

class MSE : public LossFunction
{
public:
	double Calculate(double output, double target)
	{
		return pow(output - target, 2);
	}
	double Derivative(double output, double target)
	{
		return 2 * (output - target);
	}

	std::vector<double> Calculate(std::vector<double> output, std::vector<double> target)
	{
		std::vector<double> loss(output.size());
		for (int i = 0; i < output.size(); i++)
		{
			loss[i] = Calculate(output[i], target[i]);
		}
		return loss;
	}
	std::vector<double> Derivative(std::vector<double> output, std::vector<double> target)
	{
		std::vector<double> loss(output.size());
		for (int i = 0; i < output.size(); i++)
		{
			loss[i] = Derivative(output[i], target[i]);
		}
		return loss;
	}
};

double MeanSquaredError(std::vector<double> output, std::vector<double> target)
{
	double loss = 0;
	for (int i = 0; i < output.size(); i++)
	{
		loss += pow(output[i] - target[i], 2);
	}
	return loss;
}

int main()
{
	srand(time(0));
	MSE mse;
	Valid2DConvolutionLayer Proper(3, 3);
	Valid2DConvolutionLayer c(3, 3);
	c.randomize_k(-1, 1);
	std::vector<std::vector<double>> x = {
		{1, 2, 3, 4, 5, 6},
		{7, 8, 9, 10, 11, 12},
		{13, 14, 15, 16, 17, 18},
		{19, 20, 21, 22, 23, 24},
		{25, 26, 27, 28, 29, 30},
		{31, 32, 33, 34, 35, 36}};
	
	std::vector<std::vector<double>> k = {
		{1, .5, 1},
		{.5, 1, .5},
		{1, .5, 1}};
		
	Proper.set_k(k);

	for (int i = 0; i < 10000; i++)
	{
		for (int i = 0; i < x.size(); i++)
		{
			for (int j = 0; j < x[0].size(); j++)
			{
				x[i][j] = RandRange(-10, 10);
			}
		}
		std::vector<std::vector<double>> target = Proper.forward(x, new Sigmoid());
		std::vector<std::vector<double>> y = c.forward(x, new Sigmoid());
		std::vector<std::vector<double>> y_grad;
		for (int i = 0; i < y.size(); i++)
		{
			y_grad.push_back(std::vector<double>(y[0].size(), 1));
			for (int j = 0; j < y[0].size(); j++)
			{
				y_grad[i][j] = mse.Derivative(y[i][j], target[i][j]);
			}
		}
		c.backward(new Sigmoid(), y_grad, .01);
	}

	system("clear");
	std::vector<std::vector<double>> kernel = c.get_k();
	for (int i = 0; i < kernel.size(); i++)
	{
		for (int j = 0; j < kernel[0].size(); j++)
		{
			printf("%f ", kernel[i][j]);
		}
		printf("\n");
	}

	return 0;
}