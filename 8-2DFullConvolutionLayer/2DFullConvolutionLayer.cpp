#include "2DFullConvolutionLayer.hpp"

Full2DConvolutionLayer::Full2DConvolutionLayer(int KernelXSize, int KernelYSize)
{
	k = std::vector<std::vector<double>>(KernelXSize, std::vector<double>(KernelYSize, 0));
}

std::vector<std::vector<double>> Full2DConvolutionLayer::forward(std::vector<std::vector<double>> x, ActivationFunction *af)
{
	if (x.size() % k.size() != 0 || x[0].size() % k[0].size() != 0)
	{
		throw std::invalid_argument("x dimensions must be divisible by k dimensions");
	}
	this->x = x;
	int paddingX = k.size() - 1;
	int strideX = 1;
	int paddingY = k[0].size() - 1;
	int strideY = 1;


	z = Calculate2DConvolution(x, k, strideX, strideY, paddingX, paddingY);
	std::vector<std::vector<double>> y=z;
	for (int i = 0; i < y.size(); i++)
	{
		for (int j = 0; j < y[0].size(); j++)
		{
			y[i][j] = af->Activate(y[i][j]);
		}
	}
	return y;
}

std::vector<std::vector<double>> Full2DConvolutionLayer::backward(ActivationFunction *af, std::vector<std::vector<double>> ForwardGradient, double lr)
{
	std::vector<std::vector<double>> dk=Convolution2DGetKernelGradient(x, k, ForwardGradient, 1, 1, k.size() - 1, k[0].size() - 1, af, lr);
	for (int i = 0; i < k.size(); i++)
	{
		for (int j = 0; j < k[0].size(); j++)
		{
			k[i][j] -= dk[i][j];
		}
	}
	return Convolution2DGetXGradient(x, k, ForwardGradient, 1, 1, k.size() - 1, k[0].size() - 1, af, lr);
}

std::vector<std::vector<double>> Full2DConvolutionLayer::get_x()
{
	return x;
}

std::vector<std::vector<double>> Full2DConvolutionLayer::get_k()
{
	return k;
}

void Full2DConvolutionLayer::set_k(std::vector<std::vector<double>> k)
{
	this->k = k;
}

void Full2DConvolutionLayer::randomize_k(double min, double max)
{
	for (int i = 0; i < k.size(); i++)
	{
		for (int j = 0; j < k[0].size(); j++)
		{
			k[i][j] = RandRange(min, max);
		}
	}
}