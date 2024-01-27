#include "Perceptron.hpp"

Perceptron::Perceptron() {
}

Perceptron::~Perceptron() {
}

void Perceptron::Init(int size, ActivationFunction* af, double min, double max) {
	this->af = af;
	w.resize(size);
	RandomizeWeights(min, max);
	RandomizeBias(min, max);
}

double Perceptron::Forward(std::vector<double> input) {
	x=input;
	z = 0.0;
	for (int i = 0; i < x.size(); i++) {
		z += x[i] * w[i];
	}
	z+=b;
	return af->Activate(z);
}

std::vector<double> Perceptron::Backward(double fg, double lr) {
	std::vector<double> dx(w.size());
	double dz = af->Derivative(z) * fg;
	for (int i = 0; i < w.size(); i++) {
		dx[i]=w[i] * dz;
		w[i]-=lr * dz * x[i];
	}
	b -= lr * dz;
	return dx;
}

std::vector<double> Perceptron::GetWeights() {
	return w;
}

void Perceptron::SetWeights(std::vector<double> weights) {
	w=weights;
}

void Perceptron::RandomizeWeights(double min, double max) {
	for (int i = 0; i < w.size(); i++) {
		double r = RandRange(min, max);
		while(r==0.0) r = RandRange(min, max);
		w[i] = r;
	}
}

void Perceptron::ResizeWithRandomForNewWeights(int size, double min, double max) {
	int wsize = w.size();
	w.resize(size);
	for (int i = wsize; i < w.size(); i++) {

		double r = RandRange(min, max);
		while(r==0.0) r = RandRange(min, max);
		w[i] = r;
	}
}

double Perceptron::GetBias() {
	return b;
}

void Perceptron::SetBias(double bias) {
	b = bias;
}

void Perceptron::RandomizeBias(double min, double max) {
	double r = RandRange(min, max);
	while(r==0.0) r = RandRange(min, max);
	b = r;
}