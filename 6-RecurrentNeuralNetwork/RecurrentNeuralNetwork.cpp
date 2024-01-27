#include "RecurrentNeuralNetwork.hpp"

RecurrentNeuralNetwork::RecurrentNeuralNetwork()
{
}

RecurrentNeuralNetwork::~RecurrentNeuralNetwork()
{
}

void RecurrentNeuralNetwork::Init(std::vector<int> LayerSizes, ActivationFunction *af, double min, double max)
{
	this->af = af;
	w.resize(LayerSizes.size() - 1);
	for (int i = 0; i < w.size(); i++)
	{
		w[i].resize(LayerSizes[i]+LayerSizes[i+1]);
		for (int j = 0; j < w[i].size(); j++)
		{
			w[i][j].resize(LayerSizes[i + 1]+LayerSizes[i+2]);
		}
	}

	b.resize(LayerSizes.size() - 1);
	for (int i = 0; i < b.size(); i++)
	{
		b[i].resize(LayerSizes[i + 1]);
	}

	x.resize(LayerSizes.size());
	for (int i = 0; i < x.size(); i++)
	{
		if(i==0)
			x[i].resize(LayerSizes[i]);
		else
			x[i].resize(LayerSizes[i]+LayerSizes[i+1]);
	}

	z.resize(LayerSizes.size() - 1);
	for (int i = 0; i < z.size(); i++)
	{
		z[i].resize(LayerSizes[i + 1]);
	}

	RandomizeWeights(min, max);
	RandomizeBias(min, max);
}

std::vector<double> RecurrentNeuralNetwork::ForwardLayer(std::vector<double> _x, std::vector<std::vector<double>> _w, std::vector<double> _b, std::vector<double> &_z, ActivationFunction *_af)
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

std::vector<double> RecurrentNeuralNetwork::BackwardLayer(std::vector<double> _x, std::vector<std::vector<double>> &_w, std::vector<double> &_b, std::vector<double> _z, ActivationFunction *_af, std::vector<double> _fg, double _lr)
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

std::vector<double> RecurrentNeuralNetwork::Forward(std::vector<double> input)
{
	// forward the network layer by layer using ForwardLayer
	// return the output of the last layer
	x[0] = input;
	for (int i = 0; i < previous_activation.size(); i++)
	{
		x[0].push_back(previous_activation[0][i]);
	}
	for (int i = 0; i < w.size(); i++)
	{
		x[i + 1] = ForwardLayer(x[i], w[i], b[i], z[i], af);
		x[i + 1].push_back(x[i][0]);
	}
	
	return x[w.size()];
}

std::vector<double> RecurrentNeuralNetwork::Backward(std::vector<double> fg, double lr)
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
	b.resize(w[0].size());
	for (int j = 0; j < w[0].size(); j++)
	{
		b[j].resize(w[0][j].size());
		for (int k = 0; k < w[0][j].size(); k++)
		{
			double r = RandRange(min, max);
			while (r == 0.0)
				r = RandRange(min, max);
			b[j][k] = r;
		}
	}
}
