#include "RecurrentNeuralNetwork.hpp"

RecurrentNeuralNetwork::RecurrentNeuralNetwork()
{
}

RecurrentNeuralNetwork::~RecurrentNeuralNetwork()
{
}

double RecurrentNeuralNetwork::Clamp(double value, double min, double max)
{
	if (value < min)
		return min;
	if (value > max)
		return max;
	return value;
}

double RecurrentNeuralNetwork::Sigmoid(double value, double min, double max)
{
	return (max - min) / (1.0 + exp(-value)) + min;
}

void RecurrentNeuralNetwork::Init(std::vector<int> LayerSizes, ActivationFunction *af, double min, double max)
{
	this->af = af;
	
	x.resize(LayerSizes.size());
	w.resize(LayerSizes.size() - 1);
	b.resize(LayerSizes.size() - 1);
	z.resize(LayerSizes.size() - 1);
	previous_activation.resize(LayerSizes.size() - 1);
	for (int i = 1; i < LayerSizes.size(); i++)
	{
		ResizeLayer(LayerSizes[i-1], LayerSizes[i], i-1);
	}
	x[x.size()-1].resize(LayerSizes[LayerSizes.size()-1]);

	RandomizeWeights(min, max);
	RandomizeBias(min, max);
}

void RecurrentNeuralNetwork::ResizeLayer(int InSize,int OutSize,int layer){
	w[layer].resize(InSize+OutSize);
	previous_activation[layer].resize(OutSize);
	for (int i = 0; i < w[layer].size(); i++) {
		w[layer][i].resize(OutSize);
	}
	b[layer].resize(OutSize);
	z[layer].resize(OutSize);
	x[layer].resize(InSize+OutSize);
}

std::vector<double> RecurrentNeuralNetwork::ForwardLayer(std::vector<double> _x, std::vector<std::vector<double>> _w, std::vector<double> _b, std::vector<double> &_z, std::vector<double> &_previous_activation, ActivationFunction *_af)
{
	for (int i = 0; i < _previous_activation.size(); i++) {
		_x.push_back(_previous_activation[i]);
	}
	_z.resize(_w[0].size());
	for (int j = 0; j < _w[0].size(); j++) {
		_z[j] = 0.0;
		for (int i = 0; i < _x.size(); i++) {
			_z[j] += _x[i] * _w[i][j];
		}
		_z[j]+=_b[j];
	}
	_previous_activation=_af->Activate(_z);
	return _previous_activation;
}

std::vector<double> RecurrentNeuralNetwork::BackwardLayer(std::vector<double> _x, std::vector<std::vector<double>> &_w, std::vector<double> &_b, std::vector<double> _z, std::vector<double> _previous_activation, ActivationFunction *_af, std::vector<double> _fg, double _lr)
{
	for (int i = 0; i < _previous_activation.size(); i++) {
		_x.push_back(_previous_activation[i]);
	}

	std::vector<double> dx(_w.size());
	std::vector<double> dz(_w[0].size());

	for (int j = 0; j < _w[0].size(); j++)
	{
		dz[j] = _af->Derivative(_z[j]) * _fg[j];
		for (int i = 0; i < _w.size(); i++)
		{
			dx[i] +=_lr*_w[i][j] * dz[j];
			_w[i][j] -= _lr*_x[i] * dz[j];
		}
		_b[j] -= _lr*dz[j];
	}
	
	return dx;
}

std::vector<double> RecurrentNeuralNetwork::Forward(std::vector<double> input)
{
	// forward the network layer by layer using ForwardLayer
	// return the output of the last layer
	x[0] = input;
	for (int i = 0; i < w.size(); i++)
	{
		x[i + 1] = ForwardLayer(x[i], w[i], b[i], z[i],previous_activation[i], af);
	}
	
	return x[w.size()];
}

std::vector<double> RecurrentNeuralNetwork::Backward(std::vector<double> fg, double lr)
{
	// backward the network layer by layer using BackwardLayer
	// return the output of the first layer
	for (int i = w.size() - 1; i >= 0; i--)
	{
		fg = BackwardLayer(x[i], w[i], b[i], z[i],previous_activation[i], af, fg, lr);
	}
	return fg;
}

std::vector<std::vector<std::vector<double>>> RecurrentNeuralNetwork::GetWeights()
{
	return w;
}

void RecurrentNeuralNetwork::SetWeights(std::vector<std::vector<std::vector<double>>> weights)
{
	w = weights;
}

void RecurrentNeuralNetwork::RandomizeWeights(double min, double max)
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

void RecurrentNeuralNetwork::ResizeWithRandomForNewWeights(int InSize, int OutSize, double min, double max)
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

std::vector<std::vector<double>> RecurrentNeuralNetwork::GetBias()
{
	return b;
}

void RecurrentNeuralNetwork::SetBias(std::vector<std::vector<double>> bias)
{
	b = bias;
}

void RecurrentNeuralNetwork::RandomizeBias(double min, double max)
{
	for (int j = 0; j < b.size(); j++)
	{
		for (int k = 0; k < b[j].size(); k++)
		{
			double r = RandRange(min, max);
			while (r == 0.0)
				r = RandRange(min, max);
			b[j][k] = r;
		}
	}
}

std::vector<std::vector<std::vector<double>>> RecurrentNeuralNetwork::GetW()
{
	return w;
}

std::vector<std::vector<double>> RecurrentNeuralNetwork::GetB()
{
	return b;
}

std::vector<std::vector<double>> RecurrentNeuralNetwork::GetZ()
{
	return z;
}

std::vector<std::vector<double>> RecurrentNeuralNetwork::GetX()
{
	return x;
}

std::vector<std::vector<double>> RecurrentNeuralNetwork::GetPreviousActivation()
{
	return previous_activation;
}