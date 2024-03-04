#pragma once
#include <vector>
class LossFunction
{
public:
	virtual double Calculate(double output, double target) = 0;
	virtual double Derivative(double output, double target) = 0;
	virtual std::vector<double> Calculate(std::vector<double> output, std::vector<double> target) = 0;
	virtual std::vector<double> Derivative(std::vector<double> output, std::vector<double> target) = 0;
};