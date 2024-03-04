#pragma once
#include <vector>
#include <functional>

#include "../0-Helpers/HelperFunctions.hpp"
#include "../0-Helpers/ActivationFunction.hpp"

class RecurrentPerceptron
{
private:
	std::vector<double> w,x;
	double b,z,last_activation;
	ActivationFunction* af;
public:
	RecurrentPerceptron();
	~RecurrentPerceptron();

	void Init(int InSize, ActivationFunction* af, double min=-1.0, double max=1.0);

	double Forward(std::vector<double> x);
	std::vector<double> Backward(double fg=1,double lr=0.01);

	std::vector<double> GetWeights();
	void SetWeights(std::vector<double> w);
	void RandomizeWeights(double min, double max);
	void ResizeWithRandomForNewWeights(int size, double min, double max);

	double GetBias();
	void SetBias(double b);
	void RandomizeBias(double min, double max);
};