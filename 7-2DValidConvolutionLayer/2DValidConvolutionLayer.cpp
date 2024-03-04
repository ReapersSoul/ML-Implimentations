#include "2DValidConvolutionLayer.hpp"

Valid2DConvolutionLayer::Valid2DConvolutionLayer(int KernelXSize, int KernelYSize) {
	k = std::vector<std::vector<double>>(KernelXSize, std::vector<double>(KernelYSize, 0));
}

std::vector<std::vector<double>> Valid2DConvolutionLayer::forward(std::vector<std::vector<double>> x, ActivationFunction *af) {
	if(x.size()%k.size()!=0 || x[0].size()%k[0].size()!=0) {
		throw std::invalid_argument("x dimensions must be divisible by k dimensions");
	}
	this->x = x;
	std::vector<std::vector<double>> y(x.size()-k.size()+1, std::vector<double>(x[0].size()-k[0].size()+1, 0));
	z = std::vector<std::vector<double>>(x.size()-k.size()+1, std::vector<double>(x[0].size()-k[0].size()+1, 0));
	for(int i=0; i<y.size(); i++) {
		for(int j=0; j<y[0].size(); j++) {
			for(int ii=0; ii<k.size(); ii++) {
				for(int jj=0; jj<k[0].size(); jj++) {
					z[i][j] += x[i+ii][j+jj]*k[ii][jj];
				}
			}
			y[i][j] = af->Activate(z[i][j]);
		}
	}
	return y;
}

std::vector<std::vector<double>> Valid2DConvolutionLayer::backward(ActivationFunction *af, std::vector<std::vector<double>> ForwardGradient, double lr) {
	std::vector<std::vector<double>> x_grad(x.size(), std::vector<double>(x[0].size(), 0));
	for(int i=0; i<x.size(); i++) {
		for(int j=0; j<x[0].size(); j++) {
			for(int ii=0; ii<k.size(); ii++) {
				for(int jj=0; jj<k[0].size(); jj++) {
					if(i-ii>=0 && i-ii<ForwardGradient.size() && j-jj>=0 && j-jj<ForwardGradient[0].size()) {
						x_grad[i][j] += ForwardGradient[i-ii][j-jj]*k[ii][jj]*lr*af->Derivative(z[i-ii][j-jj]);
						k[ii][jj] -= ForwardGradient[i-ii][j-jj]*x[i][j]*lr*af->Derivative(z[i-ii][j-jj]);
					}
				}
			}
		}
	}
	return x_grad;
}

std::vector<std::vector<double>> Valid2DConvolutionLayer::get_x() {
	return x;
}

std::vector<std::vector<double>> Valid2DConvolutionLayer::get_k() {
	return k;
}

void Valid2DConvolutionLayer::set_k(std::vector<std::vector<double>> k) {
	this->k = k;
}

void Valid2DConvolutionLayer::randomize_k(double min, double max) {
	for(int i=0; i<k.size(); i++) {
		for(int j=0; j<k[0].size(); j++) {
			k[i][j] = RandRange(min, max);
		}
	}
}