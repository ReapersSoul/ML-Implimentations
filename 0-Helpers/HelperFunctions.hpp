#pragma once
#include <random>
#include "ActivationFunction.hpp"

//rand range with a proper distribution
static std::random_device rd;
static std::mt19937 gen(rd());
static double RandRange(double min, double max) {
	std::uniform_real_distribution<> dis(min, max);
	return dis(gen);
}

static double CalculateConvolutionOutputSize(int inputSize, int kernelSize, int stride, int padding)
{
	return (((inputSize)+2*padding-(kernelSize-1)-1)/stride)+1;
}

static std::vector<std::vector<double>> Calculate2DConvolution(std::vector<std::vector<double>> x, std::vector<std::vector<double>> k, int strideX, int strideY,int paddingX,int paddingY)
{
	int inputSizeX = x.size();
	int inputSizeY = x[0].size();
	int kernelSizeX = k.size();
	int kernelSizeY = k[0].size();
	int outputSizeX = CalculateConvolutionOutputSize(inputSizeX, kernelSizeX, strideX, paddingX);
	int outputSizeY = CalculateConvolutionOutputSize(inputSizeY, kernelSizeY, strideY, paddingY);

	std::vector<std::vector<double>> z(outputSizeX, std::vector<double>(outputSizeY, 0));
	for (int i = outputSizeX; i >= 0; i--)
	{
		for (int j = outputSizeY; j >= 0; j--)
		{
			for (int ii = 0; ii < kernelSizeX; ii++)
			{
				for (int jj = 0; jj < kernelSizeY; jj++)
				{
					// if not out of bounds, add to z
					if (!(i*strideX + ii - paddingX < 0 || i*strideX + ii - paddingX >= inputSizeX || j*strideY + jj - paddingY < 0 || j*strideY + jj - paddingY >= inputSizeY))
					{
						z[i][j] += x[i*strideX + ii - paddingX][j*strideY + jj - paddingY] * k[ii][jj];
					}
				}
			}
		}
	}
	return z;
}

static std::vector<std::vector<double>> Convolution2DGetKernelGradient(std::vector<std::vector<double>> x, std::vector<std::vector<double>> k, std::vector<std::vector<double>> ForwardGradient, int strideX, int strideY, int paddingX, int paddingY, ActivationFunction *af, double lr)
{
	int inputSizeX = x.size();
	int inputSizeY = x[0].size();
	int kernelSizeX = k.size();
	int kernelSizeY = k[0].size();
	int outputSizeX = CalculateConvolutionOutputSize(inputSizeX, kernelSizeX, strideX, paddingX);
	int outputSizeY = CalculateConvolutionOutputSize(inputSizeY, kernelSizeY, strideY, paddingY);

	std::vector<std::vector<double>> k_grad(kernelSizeX, std::vector<double>(kernelSizeY, 0));
	for (int i = 0; i < outputSizeX; i++)
	{
		for (int j = 0; j < outputSizeY; j++)
		{
			for (int ii = 0; ii < kernelSizeX; ii++)
			{
				for (int jj = 0; jj < kernelSizeY; jj++)
				{
					// if not out of bounds, add to z
					if (!(i*strideX + ii - paddingX < 0 || i*strideX + ii - paddingX >= inputSizeX || j*strideY + jj - paddingY < 0 || j*strideY + jj - paddingY >= inputSizeY))
					{
						k_grad[ii][jj] += x[i*strideX + ii - paddingX][j*strideY + jj - paddingY] * af->Derivative(ForwardGradient[i][j])*lr;
					}
				}
			}
		}
	}
	return k_grad;
}

static std::vector<std::vector<double>> Convolution2DGetXGradient(std::vector<std::vector<double>> x, std::vector<std::vector<double>> k, std::vector<std::vector<double>> ForwardGradient, int strideX, int strideY, int paddingX, int paddingY, ActivationFunction *af, double lr)
{
	int inputSizeX = x.size();
	int inputSizeY = x[0].size();
	int kernelSizeX = k.size();
	int kernelSizeY = k[0].size();
	int outputSizeX = CalculateConvolutionOutputSize(inputSizeX, kernelSizeX, strideX, paddingX);
	int outputSizeY = CalculateConvolutionOutputSize(inputSizeY, kernelSizeY, strideY, paddingY);

	std::vector<std::vector<double>> x_grad(inputSizeX, std::vector<double>(inputSizeY, 0));
	for (int i = 0; i < outputSizeX; i++)
	{
		for (int j = 0; j < outputSizeY; j++)
		{
			for (int ii = 0; ii < kernelSizeX; ii++)
			{
				for (int jj = 0; jj < kernelSizeY; jj++)
				{
					// if not out of bounds, add to z
					if (!(i*strideX + ii - paddingX < 0 || i*strideX + ii - paddingX >= inputSizeX || j*strideY + jj - paddingY < 0 || j*strideY + jj - paddingY >= inputSizeY))
					{
						x_grad[i*strideX + ii - paddingX][j*strideY + jj - paddingY] += k[ii][jj] * af->Derivative(ForwardGradient[i][j])*lr;
					}
				}
			}
		}
	}
	
	return x_grad;
}
