#include "../Headers/MLLib.hpp"

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
//    // Check if kernel dimensions are compatible with the tensor
//     if (kernel.getDimensions() > this->getDimensions()) {
//         throw std::invalid_argument("Kernel dimensions are greater than tensor dimensions!");
//     }

//     // Initialize output tensor
//     std::vector<int> outputDimensions = this->calculateOutputDimensions(kernel, Stride, Padding);
//     Tensor output(outputDimensions);

//     // Iterate over the tensor
//     std::vector<int> TensorPos(this->getDimensions(), 0);
//     do {
//         // Extract block from tensor
//         Tensor block = this->Block(TensorPos, kernel.shape);

//         // Multiply block with kernel and sum the result
//         float convResult = block.MulSum(kernel);

//         // Store result in output tensor
//         output.set(TensorPos, convResult);
//     } while (this->nextPosition(TensorPos, Stride));

//     return output;
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

double Sigmoid::Activate(double x){
	return 1/(1+exp(-x));
}

double Sigmoid::Derivative(double x){
	return x*(1-x);
}

std::vector<double> Sigmoid::Activate(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_Sigmoid(x, result);
	return result;
}

std::vector<double> Sigmoid::Derivative(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_Sigmoid_Derivative(x, result);
	return result;
}

double ReLU::Activate(double x){
	return x > 0 ? x : 0;
}

double ReLU::Derivative(double x){
	return x > 0 ? 1 : 0;
}

std::vector<double> ReLU::Activate(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_ReLU(x, result);
	return result;
}

std::vector<double> ReLU::Derivative(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_ReLU_Derivative(x, result);
	return result;
}

double LeakyReLU::Activate(double x){
	return x > 0 ? x : alpha*x;
}

double LeakyReLU::Derivative(double x){
	return x > 0 ? 1 : alpha;
}

std::vector<double> LeakyReLU::Activate(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_LeakyReLU(x, result, alpha);
	return result;
}

std::vector<double> LeakyReLU::Derivative(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_LeakyReLU_Derivative(x, result, alpha);
	return result;
}

double Tanh::Activate(double x){
	return tanh(x);
}

double Tanh::Derivative(double x){
	return 1 - x*x;
}

std::vector<double> Tanh::Activate(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_Tanh(x, result);
	return result;
}

std::vector<double> Tanh::Derivative(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_Tanh_Derivative(x, result);
	return result;
}

double Softmax::Activate(double x){
	return exp(x);
}

double Softmax::Derivative(double x){
	return x*(1-x);
}

std::vector<double> Softmax::Activate(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_Softmax(x, result);
	return result;
}

std::vector<double> Softmax::Derivative(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_Softmax_Derivative(x, result);
	return result;
}

double GeLU::Activate(double x){
	return 0.5*x*(1+erf(x/sqrt(2)));
}

double GeLU::Derivative(double x){
	return 0.5*(1+erf(x/sqrt(2)) + x*exp(-x*x/2)/sqrt(2*M_PI));
}

std::vector<double> GeLU::Activate(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_GeLU(x, result);
	return result;
}

std::vector<double> GeLU::Derivative(std::vector<double> x){
	std::vector<double> result(x.size());
	GPU_GeLU_Derivative(x, result);
	return result;
}

double MeanSquaredError::Calculate(double output, double target){
	return 0.5*(output-target)*(output-target);
}

double MeanSquaredError::Derivative(double output, double target){
	return output-target;
}

std::vector<double> MeanSquaredError::Calculate(std::vector<double> output, std::vector<double> target){
	std::vector<double> result(output.size());
	GPU_MeanSquaredError(output, target, result);
	return result;
}

std::vector<double> MeanSquaredError::Derivative(std::vector<double> output, std::vector<double> target){
	std::vector<double> result(output.size());
	GPU_MeanSquaredError_Derivative(output, target, result);
	return result;
}

double MeanAbsoluteError::Calculate(double output, double target){
	return abs(output-target);
}

double MeanAbsoluteError::Derivative(double output, double target){
	return output-target > 0 ? 1 : -1;
}

std::vector<double> MeanAbsoluteError::Calculate(std::vector<double> output, std::vector<double> target){
	std::vector<double> result(output.size());
	GPU_MeanAbsoluteError(output, target, result);
	return result;
}

std::vector<double> MeanAbsoluteError::Derivative(std::vector<double> output, std::vector<double> target){
	std::vector<double> result(output.size());
	GPU_MeanAbsoluteError_Derivative(output, target, result);
	return result;
}

double LogLoss::Calculate(double output, double target){
	return -target*log(output) - (1-target)*log(1-output);
}

double LogLoss::Derivative(double output, double target){
	return (output-target)/(output*(1-output));
}

std::vector<double> LogLoss::Calculate(std::vector<double> output, std::vector<double> target){
	std::vector<double> result(output.size());
	GPU_LogLoss(output, target, result);
	return result;
}

std::vector<double> LogLoss::Derivative(std::vector<double> output, std::vector<double> target){
	std::vector<double> result(output.size());
	GPU_LogLoss_Derivative(output, target, result);
	return result;
}

double CrossEntropy::Calculate(double output, double target){
	return -target*log(output);
}

double CrossEntropy::Derivative(double output, double target){
	return -target/output;
}

std::vector<double> CrossEntropy::Calculate(std::vector<double> output, std::vector<double> target){
	std::vector<double> result(output.size());
	GPU_CrossEntropy(output, target, result);
	return result;
}

std::vector<double> CrossEntropy::Derivative(std::vector<double> output, std::vector<double> target){
	std::vector<double> result(output.size());
	GPU_CrossEntropy_Derivative(output, target, result);
	return result;
}