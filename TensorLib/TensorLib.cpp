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
   // Check if kernel dimensions are compatible with the tensor
    if (kernel.getDimensions() > this->getDimensions()) {
        throw std::invalid_argument("Kernel dimensions are greater than tensor dimensions!");
    }

    // Initialize output tensor
    std::vector<int> outputDimensions = this->calculateOutputDimensions(kernel, Stride, Padding);
    Tensor output(outputDimensions);

    // Iterate over the tensor
    std::vector<int> TensorPos(this->getDimensions(), 0);
    do {
        // Extract block from tensor
        Tensor block = this->Block(TensorPos, kernel.shape);

        // Multiply block with kernel and sum the result
        float convResult = block.MulSum(kernel);

        // Store result in output tensor
        output.set(TensorPos, convResult);
    } while (this->nextPosition(TensorPos, Stride));

    return output;
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

void Tensor::MakeIndexTensor(){
	for (int i = 0; i < data.size(); i++)
	{
		data[i] = i;
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