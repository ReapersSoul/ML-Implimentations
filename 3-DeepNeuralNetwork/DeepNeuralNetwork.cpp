#include "DeepNeuralNetwork.hpp"

DeepNeuralNetwork::DeepNeuralNetwork()
{
}

DeepNeuralNetwork::~DeepNeuralNetwork()
{
}

void DeepNeuralNetwork::Init(std::vector<int> LayerSizes, ActivationFunction *af, double min, double max)
{
	this->af = af;
	x.resize(LayerSizes.size());
	for (int i = 0; i < LayerSizes.size(); i++)
	{
		x[i].resize(LayerSizes[i]);
	}
	z.resize(LayerSizes.size() - 1);
	w.resize(LayerSizes.size() - 1);
	b.resize(LayerSizes.size() - 1);
	for (int i = 1; i < LayerSizes.size(); i++)
	{
		w[i - 1].resize(LayerSizes[i - 1]);
		b[i - 1].resize(LayerSizes[i]);
		z[i - 1].resize(LayerSizes[i]);
		for (int j = 0; j < LayerSizes[i - 1]; j++)
		{
			w[i - 1][j].resize(LayerSizes[i]);
		}
	}
	

	RandomizeWeights(min, max);
	RandomizeBias(min, max);
}

std::vector<double> DeepNeuralNetwork::ForwardLayer(std::vector<double> _x, std::vector<std::vector<double>> _w, std::vector<double> _b, std::vector<double> &_z, ActivationFunction *_af)
{
	_z.resize(_w[0].size());
	for (int j = 0; j < _w[0].size(); j++) {
		_z[j] = 0.0;
		for (int i = 0; i < _x.size(); i++) {
			_z[j] += _x[i] * _w[i][j];
		}
		_z[j]+=_b[j];
	}
	return _af->Activate(_z);
}

std::vector<double> DeepNeuralNetwork::BackwardLayer(std::vector<double> _x, std::vector<std::vector<double>> &_w, std::vector<double> &_b, std::vector<double> _z, ActivationFunction *_af, std::vector<double> _fg, double _lr)
{
	std::vector<double> dx(_w.size());
	std::vector<double> dz(_w[0].size());
	for (int j = 0; j < _w[0].size(); j++)
	{
		dz[j] = _af->Derivative(_z[j]) * _fg[j];
		for (int i = 0; i < _w.size(); i++)
		{
			dx[i] += _w[i][j] * dz[j];
			_w[i][j] -= _lr * dz[j] * _x[i];
		}
		_b[j] -= _lr * dz[j];
	}
	return dx;
}

std::vector<double> DeepNeuralNetwork::Forward(std::vector<double> input)
{
	// forward the network layer by layer using ForwardLayer
	// return the output of the last layer
	x[0] = input;
	for (int i = 0; i < w.size(); i++)
	{
		x[i + 1] = ForwardLayer(x[i], w[i], b[i], z[i], af);
	}
	
	return x[w.size()];
}

std::vector<double> DeepNeuralNetwork::Backward(std::vector<double> fg, double lr)
{
	// backward the network layer by layer using BackwardLayer
	// return the output of the first layer
	std::vector<std::thread> threads;
	for (int i = w.size() - 1; i >= 0; i--)
	{
		fg = BackwardLayer(x[i], w[i], b[i], z[i], af, fg, lr);
	}
	return fg;
}

std::vector<std::vector<std::vector<double>>> DeepNeuralNetwork::GetWeights()
{
	return w;
}

void DeepNeuralNetwork::SetWeights(std::vector<std::vector<std::vector<double>>> weights)
{
	w = weights;
}

void DeepNeuralNetwork::RandomizeWeights(double min, double max)
{
	for (int i = 0; i < w.size(); i++)
	{
		for (int j = 0; j < w[i].size(); j++)
		{
			for (int k = 0; k < w[i][j].size(); k++)
			{
				double r = RandRange(min, max);
				while (r == 0.0)
					r = RandRange(min, max);
				w[i][j][k] = r;
			}
		}
	}
}

void DeepNeuralNetwork::ResizeWithRandomForNewWeights(int InSize, int OutSize, double min, double max)
{
	int wsize = w.size();
	w.resize(InSize);
	for (int i = wsize; i < w.size(); i++)
	{
		w[i].resize(OutSize);
		for (int j = 0; j < w[i].size(); j++)
		{
			w[i][j].resize(OutSize);
			for (int k = 0; k < w[i][j].size(); k++)
			{
				double r = RandRange(min, max);
				while (r == 0.0)
					r = RandRange(min, max);
				w[i][j][k] = r;
			}
		}
	}
}

std::vector<std::vector<double>> DeepNeuralNetwork::GetBias()
{
	return b;
}

void DeepNeuralNetwork::SetBias(std::vector<std::vector<double>> bias)
{
	b = bias;
}

void DeepNeuralNetwork::RandomizeBias(double min, double max)
{
	for (int i = 0; i < b.size(); i++)
	{
		for (int j = 0; j < b[i].size(); j++)
		{
			double r = RandRange(min, max);
			while (r == 0.0)
				r = RandRange(min, max);
			b[i][j] = r;
		}
	}
}

std::vector<std::vector<std::vector<double>>> DeepNeuralNetwork::GetW()
{
	return w;
}

std::vector<std::vector<double>> DeepNeuralNetwork::GetB()
{
	return b;
}

std::vector<std::vector<double>> DeepNeuralNetwork::GetZ()
{
	return z;
}

std::vector<std::vector<double>> DeepNeuralNetwork::GetX()
{
	return x;
}