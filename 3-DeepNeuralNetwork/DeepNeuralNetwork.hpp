#pragma once
#include <vector>
#include <functional>
#include <thread>

#include "../0-Helpers/HelperFunctions.hpp"
#include "../0-Helpers/ActivationFunction.hpp"

class DeepNeuralNetwork
{
private:
	std::vector<std::vector<std::vector<double>>> w;
	std::vector<std::vector<double>> b, z, x;
	ActivationFunction *af;

public:
	DeepNeuralNetwork();
	~DeepNeuralNetwork();

	void Init(std::vector<int> LayerSizes, ActivationFunction *af, double min = -1.0, double max = 1.0);

	static std::vector<double> ForwardLayer(std::vector<double> x, std::vector<std::vector<double>> w, std::vector<double> b, std::vector<double> &z, ActivationFunction *af);
	static std::vector<double> BackwardLayer(std::vector<double> x, std::vector<std::vector<double>> &w, std::vector<double> &b, std::vector<double> z, ActivationFunction *af, std::vector<double> fg=std::vector<double>(1), double lr=.01);

	std::vector<double> Forward(std::vector<double> x);
	std::vector<double> Backward(std::vector<double> fg = std::vector<double>(1), double lr = 0.01);

	std::vector<std::vector<std::vector<double>>> GetWeights();
	void SetWeights(std::vector<std::vector<std::vector<double>>> w);
	void RandomizeWeights(double min, double max);
	void ResizeWithRandomForNewWeights(int InSize, int OutSize, double min, double max);

	std::vector<std::vector<double>> GetBias();
	void SetBias(std::vector<std::vector<double>> b);
	void RandomizeBias(double min, double max);
};