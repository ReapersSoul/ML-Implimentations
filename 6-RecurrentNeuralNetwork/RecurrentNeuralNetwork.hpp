#pragma once
#include <vector>
#include <functional>
#include <thread>

#include "../0-Helpers/HelperFunctions.hpp"
#include "../0-Helpers/ActivationFunction.hpp"

class RecurrentNeuralNetwork
{
private:
	std::vector<std::vector<std::vector<double>>> w;
	std::vector<std::vector<double>> b, z, x, previous_activation;
	ActivationFunction *af;

	static double Clamp(double value, double min, double max);

	static double Sigmoid(double value, double min, double max);

public:
	RecurrentNeuralNetwork();
	~RecurrentNeuralNetwork();

	void Init(std::vector<int> LayerSizes, ActivationFunction *af, double min = -1.0, double max = 1.0);
	void ResizeLayer(int InSize,int OutSize,int layer);

	static std::vector<double> ForwardLayer(std::vector<double> x, std::vector<std::vector<double>> w, std::vector<double> b, std::vector<double> &z, std::vector<double> &previous_activation, ActivationFunction *af);
	static std::vector<double> BackwardLayer(std::vector<double> _x, std::vector<std::vector<double>> &_w, std::vector<double> &_b, std::vector<double> _z, std::vector<double> _previous_activation, ActivationFunction *_af, std::vector<double> fg=std::vector<double>(1), double lr=.01);

	std::vector<double> Forward(std::vector<double> x);
	std::vector<double> Backward(std::vector<double> fg = std::vector<double>(1), double lr = 0.01);

	std::vector<std::vector<std::vector<double>>> GetWeights();
	void SetWeights(std::vector<std::vector<std::vector<double>>> w);
	void RandomizeWeights(double min, double max);
	void ResizeWithRandomForNewWeights(int InSize, int OutSize, double min, double max);

	std::vector<std::vector<double>> GetBias();
	void SetBias(std::vector<std::vector<double>> b);
	void RandomizeBias(double min, double max);

	//getters
	std::vector<std::vector<std::vector<double>>> GetW();
	std::vector<std::vector<double>> GetB();
	std::vector<std::vector<double>> GetZ();
	std::vector<std::vector<double>> GetX();
	std::vector<std::vector<double>> GetPreviousActivation();
};