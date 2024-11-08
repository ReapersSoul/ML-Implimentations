#include "PerceptronLayer.hpp"

PerceptronLayer::PerceptronLayer() {
}

PerceptronLayer::~PerceptronLayer() {
}

void PerceptronLayer::Init(int InSize, int OutSize, ActivationFunction* af, double min, double max) {
	this->af = af;
	w.resize(InSize);
	for (int i = 0; i < w.size(); i++) {
		w[i].resize(OutSize);
	}
	RandomizeWeights(min, max);
	RandomizeBias(min, max);
}

std::vector<double> PerceptronLayer::Forward(std::vector<double> input) {
	x=input;
	z.resize(w[0].size());
	for (int j = 0; j < w[0].size(); j++) {
		z[j] = 0.0;
		GPU_MulSum(x, w[j], z[j]);
		z[j]+=b[j];
	}
	return af->Activate(z);
}

std::vector<double> PerceptronLayer::Backward(std::vector<double> fg, double lr) {
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

std::vector<double> PerceptronLayer::GetWeights() {
	std::vector<double> weights;
	for (int i = 0; i < w.size(); i++) {
		for (int j = 0; j < w[i].size(); j++) {
			weights.push_back(w[i][j]);
		}
	}
	return weights;
}

void PerceptronLayer::SetWeights(std::vector<std::vector<double>> weights) {
	w=weights;
}

void PerceptronLayer::RandomizeWeights(double min, double max) {
	for (int i = 0; i < w.size(); i++) {
		for (int j = 0; j < w[i].size(); j++) {
			double r = RandRange(min, max);
			while(r==0.0) r = RandRange(min, max);
			w[i][j] = r;
		}
	}
}

void PerceptronLayer::ResizeWithRandomForNewWeights(int InSize,int OutSize, double min, double max) {
	int wsize = w.size();
	w.resize(InSize);
	for (int i = wsize; i < w.size(); i++) {
		w[i].resize(OutSize);
		for (int j = 0; j < w[i].size(); j++) {
			double r = RandRange(min, max);
			while(r==0.0) r = RandRange(min, max);
			w[i][j] = r;
		}
	}
}

std::vector<double> PerceptronLayer::GetBias() {
	return b;
}

void PerceptronLayer::SetBias(std::vector<double> bias) {
	b=bias;
}

void PerceptronLayer::RandomizeBias(double min, double max) {
	b.resize(w[0].size());
	for (int j = 0; j < w[0].size(); j++) {
		double r = RandRange(min, max);
		while(r==0.0) r = RandRange(min, max);
		b[j] = r;
	}
}
