#include "TensorLib.hpp"

Tensor::Tensor(std::vector<int> Shape){
	this->shape = Shape;
	int size = 1;
	for(int i = 0; i < Shape.size(); i++){
		size *= Shape[i];
	}
	data.resize(size,1);
}

Tensor::~Tensor(){
}

int Tensor::getDimensions(){
	return shape.size();
}

std::vector<int> Tensor::getShape(){
	return shape;
}

Tensor Tensor::Block(std::vector<int> Start, std::vector<int> Shape){
	//get the number of arguments
	if(Shape.size()>shape.size()){
		throw std::invalid_argument("The number of dimensions of the block is greater than the number of dimensions of the tensor");
	}
	
	Tensor t(Shape);
	Tensor_Block(this->data, this->shape, Start, Shape, t.data);

	return t;
}

Tensor Tensor::Transpose(){
	//TODO
}

Tensor Tensor::Dot(Tensor &t){
	//TODO
}

Tensor Tensor::Conv(Tensor &t, int Stride, int Padding){
	//TODO
}

double Tensor::MulSum(Tensor &t){
	double sum = 0;
	Tensor_MulSum(this->data, t.data, sum);
	return sum;
}

double Tensor::Sum(){
	double sum = 0;
	Tensor_Sum(this->data, sum);
	return sum;
}

void Tensor::Randomize(double min, double max){
	for(int i = 0; i < data.size(); i++){
		data[i] = randRange(-1,1);
	}
}

void Tensor::Print(){
	//use the shape and data to print the tensor
	printf("Shape: ");
	for(int i = 0; i < shape.size(); i++){
		printf("%d ", shape[i]);
	}
	printf("\n");

	printf("Data: ");
	for(int i = 0; i < data.size(); i++){
		printf("%f ", data[i]);
	}
	printf("\n");


}