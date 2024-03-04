#pragma once
#include <vector>
#include <stdexcept>
#include "../0-Helpers/HelperFunctions.hpp"
#include "../0-Helpers/ActivationFunction.hpp"

class Valid2DConvolutionLayer {
	std::vector<std::vector<double>> x,k, z;
public:
	Valid2DConvolutionLayer(int KernelXSize, int KernelYSize);
	std::vector<std::vector<double>> forward(std::vector<std::vector<double>> x, ActivationFunction *af);
	std::vector<std::vector<double>> backward(ActivationFunction *af, std::vector<std::vector<double>> ForwardGradient=std::vector<std::vector<double>>(1), double lr=.01);
	std::vector<std::vector<double>> get_x();
	std::vector<std::vector<double>> get_k();
	void set_k(std::vector<std::vector<double>> k);
	void randomize_k(double min, double max);
	void resize_with_random_for_new_k(int KernelXSize, int KernelYSize, double min, double max);
};