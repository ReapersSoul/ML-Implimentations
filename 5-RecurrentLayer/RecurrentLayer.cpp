#include "RecurrentLayer.hpp"

RecurrentLayer::RecurrentLayer() {
}

RecurrentLayer::~RecurrentLayer() {
}

void RecurrentLayer::Init(int InSize, int OutSize, ActivationFunction* af, double min, double max) {
	this->af = af;
	w.resize(InSize+OutSize);
	previous_activation.resize(OutSize);
	for (int i = 0; i < w.size(); i++) {
		w[i].resize(OutSize);
	}
	RandomizeWeights(min, max);
	RandomizeBias(min, max);
}

std::vector<double> RecurrentLayer::Forward(std::vector<double> input) {
	for (int i = 0; i < previous_activation.size(); i++) {
		input.push_back(previous_activation[i]);
	}	
	x=input;
	z.resize(w[0].size());
	for (int j = 0; j < w[0].size(); j++) {
		z[j] = 0.0;
		for (int i = 0; i < x.size(); i++) {
			z[j] += x[i] * w[i][j];
		}
		z[j]+=b[j];
	}
	return af->Activate(z);
}

std::vector<double> RecurrentLayer::Backward(std::vector<double> fg, double lr) {
	std::vector<double> dx(w.size());
	std::vector<double> dz(w[0].size());
	for (int j = 0; j < w[0].size(); j++) {
		dz[j] = af->Derivative(z[j]) * fg[j];
		for (int i = 0; i < w.size(); i++) {
			dx[i]+=w[i][j] * dz[j];
			w[i][j]-=lr * dz[j] * x[i];
		}
		b[j] -= lr * dz[j];
	}
	return dx;
}

std::vector<double> RecurrentLayer::GetWeights() {
	std::vector<double> weights;
	for (int i = 0; i < w.size(); i++) {
		for (int j = 0; j < w[i].size(); j++) {
			weights.push_back(w[i][j]);
		}
	}
	return weights;
}

void RecurrentLayer::SetWeights(std::vector<std::vector<double>> weights) {
	w=weights;
}

void RecurrentLayer::RandomizeWeights(double min, double max) {
	for (int i = 0; i < w.size(); i++) {
		for (int j = 0; j < w[i].size(); j++) {
			double r = RandRange(min, max);
			while(r==0.0) r = RandRange(min, max);
			w[i][j] = r;
		}
	}
}

void RecurrentLayer::ResizeWithRandomForNewWeights(int InSize,int OutSize, double min, double max) {
	int wsize = w.size();
	w.resize(InSize+OutSize);
	for (int i = wsize; i < w.size(); i++) {
		w[i].resize(OutSize);
		for (int j = 0; j < w[i].size(); j++) {
			double r = RandRange(min, max);
			while(r==0.0) r = RandRange(min, max);
			w[i][j] = r;
		}
	}
}

std::vector<double> RecurrentLayer::GetBias() {
	return b;
}

void RecurrentLayer::SetBias(std::vector<double> bias) {
	b=bias;
}

void RecurrentLayer::RandomizeBias(double min, double max) {
	b.resize(w[0].size());
	for (int j = 0; j < w[0].size(); j++) {
		double r = RandRange(min, max);
		while(r==0.0) r = RandRange(min, max);
		b[j] = r;
	}
}
