#include "RecurrentPerceptron.hpp"

RecurrentPerceptron::RecurrentPerceptron() {
}

RecurrentPerceptron::~RecurrentPerceptron() {
}

void RecurrentPerceptron::Init(int size, ActivationFunction* af, double min, double max) {
	this->af = af;
	last_activation = 0.0;
	w.resize(size+1);
	RandomizeWeights(min, max);
	RandomizeBias(min, max);
}

double RecurrentPerceptron::Forward(std::vector<double> input) {
	input.push_back(last_activation);
	x=input;
	z = 0.0;
	for (int i = 0; i < x.size(); i++) {
		z += x[i] * w[i];
	}
	z+=b;
	last_activation=af->Activate(z);
	return last_activation;
}

std::vector<double> RecurrentPerceptron::Backward(double fg, double lr) {
	std::vector<double> dx(w.size());
	double dz = af->Derivative(z) * fg;
	for (int i = 0; i < w.size(); i++) {
		dx[i]=w[i] * dz;
		w[i]-=lr * dz * x[i];
	}
	b -= lr * dz;
	return dx;
}

std::vector<double> RecurrentPerceptron::GetWeights() {
	return w;
}

void RecurrentPerceptron::SetWeights(std::vector<double> weights) {
	w=weights;
}

void RecurrentPerceptron::RandomizeWeights(double min, double max) {
	for (int i = 0; i < w.size(); i++) {
		double r = RandRange(min, max);
		while(r==0.0) r = RandRange(min, max);
		w[i] = r;
	}
}

void RecurrentPerceptron::ResizeWithRandomForNewWeights(int size, double min, double max) {
	int wsize = w.size();
	w.resize(size);
	for (int i = wsize; i < w.size(); i++) {

		double r = RandRange(min, max);
		while(r==0.0) r = RandRange(min, max);
		w[i] = r;
	}
}

double RecurrentPerceptron::GetBias() {
	return b;
}

void RecurrentPerceptron::SetBias(double bias) {
	b = bias;
}

void RecurrentPerceptron::RandomizeBias(double min, double max) {
	double r = RandRange(min, max);
	while(r==0.0) r = RandRange(min, max);
	b = r;
}