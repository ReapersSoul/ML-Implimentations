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

std::vector<double> Tensor::getData(){
	return data;
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

Tensor Tensor::Conv(Tensor &kernel, int Stride, int Padding){
	std::vector<int> newShape = this->shape;
	Tensor ret(newShape);

	//use Block and MulSum to implement this
	// for(int d = 0; d < shape.size(); d++){
	// 	for(int i = 0; i < shape[d]; i++){
	// 		for(int j = 0; j < shape[d+1]; j++){
	// 			std::vector<int> start = {i,j};
	// 			std::vector<int> shape = kernel.getShape();
	// 			Tensor block = this->Block(start, shape);
	// 			double sum = block.MulSum(kernel);
	// 			this->data[posToIndex(start, this->shape, this->shape.size())] = sum;
	// 		}
	// 	}
	// }
}

double Tensor::MulSum(Tensor &t){
	double sum = 0;
	GPU_MulSum(this->data, t.data, sum);
	return sum;
}

double Tensor::Sum(){
	double sum = 0;
	GPU_Sum(this->data, sum);
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