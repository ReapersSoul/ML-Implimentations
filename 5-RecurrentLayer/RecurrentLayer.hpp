#pragma once
#include <vector>
#include <functional>

#include "../0-Helpers/HelperFunctions.hpp"
#include "../0-Helpers/ActivationFunction.hpp"

class RecurrentLayer
{
private:
	std::vector<std::vector<double>> w;
	std::vector<double> b,z,x,previous_activation;
	ActivationFunction* af;
public:
	RecurrentLayer();
	~RecurrentLayer();

	void Init(int InSize, int OutSize, ActivationFunction* af, double min=-1.0, double max=1.0);

	std::vector<double> Forward(std::vector<double> x);
	std::vector<double> Backward(std::vector<double> fg=std::vector<double>(1),double lr=0.01);

	std::vector<std::vector<double>> GetWeights();
	void SetWeights(std::vector<std::vector<double>> w);
	void RandomizeWeights(double min, double max);
	void ResizeWithRandomForNewWeights(int InSize, int OutSize, double min, double max);

	std::vector<double> GetBias();
	void SetBias(std::vector<double> b);
	void RandomizeBias(double min, double max);

	std::vector<double> GetX();
	std::vector<double> GetZ();
	std::vector<double> GetPreviousActivation();
};