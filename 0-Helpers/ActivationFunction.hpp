#pragma once
#include <vector>
class ActivationFunction
{
public:
	virtual double Activate(double input) = 0;
	virtual double Derivative(double input) = 0;
	virtual std::vector<double> Activate(std::vector<double> input) = 0;
	virtual std::vector<double> Derivative(std::vector<double> input) = 0;
};