#pragma once
#include <stdexcept>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <filesystem>
#include <CL/cl2.hpp>

static std::vector<cl::Platform> platforms;
static std::vector<cl::Device> devices;
static cl::Context context;
static cl::CommandQueue queue;
static bool isInitialized = false;
static int maxWorkGroupSize;

// misc kernels and programs
static cl::Kernel Sum_Step_Kernel;
static cl::Program Sum_Step_Program;

static cl::Kernel Mul_Kernel;
static cl::Program Mul_Program;

// tensor kernels and programs
static cl::Kernel Tensor_Block_Kernel;
static cl::Program Tensor_Block_Program;

static cl::Kernel Tensor_Transpose_Kernel;
static cl::Program Tensor_Transpose_Program;

static cl::Kernel Tensor_Dot_Kernel;
static cl::Program Tensor_Dot_Program;

static cl::Kernel Tensor_Conv_Kernel;
static cl::Program Tensor_Conv_Program;

static std::string slurp(std::ifstream &in)
{
	std::ostringstream sstr;
	sstr << in.rdbuf();
	return sstr.str();
}

static void PrintClBuildError(std::string KernelName,cl_int err, cl::Program &Program, cl::Device &Device,int line){
		char *buff_erro;
		cl_int errcode;
		size_t build_log_len;
		errcode = clGetProgramBuildInfo(Program.get(), devices[0].get(), CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
		if (errcode)
		{
			printf("clGetProgramBuildInfo failed at line %d\n", line);
			throw std::runtime_error("clGetProgramBuildInfo failed");
		}

		buff_erro = (char*)malloc(build_log_len);
		if (!buff_erro)
		{
			printf("malloc failed at line %d\n", line);
			throw std::runtime_error("malloc failed");
		}

		errcode = clGetProgramBuildInfo(Program.get(), devices[0].get(), CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
		if (errcode)
		{
			printf("clGetProgramBuildInfo failed at line %d\n", line);
			throw std::runtime_error("clGetProgramBuildInfo failed");
		}

		fprintf(stderr, "Build log for %s:\n%s\n", "Tensor_Block", buff_erro); // Be careful with  the fprint
		free(buff_erro);
		fprintf(stderr, "clBuildProgram failed\n");
}

static void InitializeOpenCL()
{
	if (isInitialized)
	{
		return;
	}
	cl::Platform::get(&platforms);
	if (platforms.empty())
	{
		throw std::runtime_error("No OpenCL platforms found!");
	}

	platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
	if (devices.empty())
	{
		throw std::runtime_error("No GPU devices found!");
	}

	// get the max work group size
	maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	
	char ext_str[8192];
	size_t ext_str_len;

	//setup context and queue for printing to console
	context = cl::Context(devices);

	queue = cl::CommandQueue(context, devices[0]);

	//get device extensions
	clGetDeviceInfo(devices[0](), CL_DEVICE_EXTENSIONS, 8192, ext_str, &ext_str_len);
	printf("Device extensions: %s\n", ext_str);

	// misc kernels
	cl::Program::Sources sources;
	std::string kernel_code;
	std::string kernel_path = "Kernels/Sum_Step.cl";
	// read whole file into string
	std::ifstream file(kernel_path);
	if (!file.is_open())
	{
		// get absolute path
		std::filesystem::path p(kernel_path);
		// print the absolute of the kernel path
		printf("Could not open file: %s\n", std::filesystem::absolute(p).string().c_str());
		// print current directory
		p = std::filesystem::current_path();
		printf("Current path: %s\n", p.string().c_str());

		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Sum_Step_Program = cl::Program(context, sources);
	cl_int err = Sum_Step_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Sum_Step",err,Sum_Step_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Sum_Step_Kernel = cl::Kernel(Sum_Step_Program, "Sum_Step");

	kernel_path = "Kernels/Mul.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Mul_Program = cl::Program(context, sources);
	err = Mul_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Mul",err,Mul_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Mul_Kernel = cl::Kernel(Mul_Program, "Mul", &err);

	// Tensor kernels
	kernel_path = "Kernels/Tensor_Block.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Tensor_Block_Program = cl::Program(context, sources);
	err = Tensor_Block_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Tensor_Block",err,Tensor_Block_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Tensor_Block_Kernel = cl::Kernel(Tensor_Block_Program, "Tensor_Block");

	kernel_path = "Kernels/Tensor_Transpose.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Tensor_Transpose_Program = cl::Program(context, sources);
	err = Tensor_Transpose_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Tensor_Transpose",err,Tensor_Transpose_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Tensor_Transpose_Kernel = cl::Kernel(Tensor_Transpose_Program, "Tensor_Transpose");

	kernel_path = "Kernels/Tensor_Dot.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Tensor_Dot_Program = cl::Program(context, sources);
	err = Tensor_Dot_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Tensor_Dot",err,Tensor_Dot_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Tensor_Dot_Kernel = cl::Kernel(Tensor_Dot_Program, "Tensor_Dot");

	kernel_path = "Kernels/Tensor_Conv.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Tensor_Conv_Program = cl::Program(context, sources);
	err = Tensor_Conv_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Tensor_Conv",err,Tensor_Conv_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Tensor_Conv_Kernel = cl::Kernel(Tensor_Conv_Program, "Tensor_Conv");

	isInitialized = true;
}

// misc functions
static void GPU_Sum_Step(std::vector<double> data, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	// create the buffers
	cl::Buffer dataBuffer(context, CL_MEM_READ_ONLY, data.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, (data.size() / 2) * sizeof(double));

	// write the data to the buffers
	queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, data.size() * sizeof(double), data.data());

	// set the arguments
	Sum_Step_Kernel.setArg(0, dataBuffer);
	Sum_Step_Kernel.setArg(1, resultBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(Sum_Step_Kernel, cl::NullRange, cl::NDRange(data.size() / 2), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, (data.size() / 2) * sizeof(double), result.data());

	// if the result is not a power of 2, add the last element
	if (data.size() % 2 != 0)
	{
		result.push_back(data[data.size() - 1]);
	}
}

static void GPU_Sum(std::vector<double> data, double &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	std::vector<double> temp;
	temp = data;
	while (temp.size() > 1)
	{
		std::vector<double> temp2;
		temp2.resize(temp.size() / 2);
		GPU_Sum_Step(temp, temp2);
		temp = temp2;
	}
	result = temp[0];
}

static void GPU_Mul(std::vector<double> data1, std::vector<double> data2, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	// check if the size of the data1 is equal to the size of the data2
	if (data1.size() != data2.size())
	{
		throw std::invalid_argument("The size of the data1 is not equal to the size of the data2");
	}

	int size = result.size();

	// create the buffers
	cl::Buffer data1Buffer(context, CL_MEM_READ_ONLY, size * sizeof(double));
	cl::Buffer data2Buffer(context, CL_MEM_READ_ONLY, size * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(double));

	// write the data to the buffers
	queue.enqueueWriteBuffer(data1Buffer, CL_TRUE, 0, size * sizeof(double), data1.data());
	queue.enqueueWriteBuffer(data2Buffer, CL_TRUE, 0, size * sizeof(double), data2.data());

	// set the arguments
	Mul_Kernel.setArg(0, data1Buffer);
	Mul_Kernel.setArg(1, data2Buffer);
	Mul_Kernel.setArg(2, resultBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(Mul_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_MulSum(std::vector<double> tensor1, std::vector<double> tensor2, double &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	std::vector<double> temp;
	temp.resize(tensor1.size());
	GPU_Mul(tensor1, tensor2, temp);
	GPU_Sum(temp, result);
}

// Tensor functions
static void Tensor_Block(std::vector<double> Tensor,std::vector<int> TensorShape,std::vector<int> BlockStart,std::vector<int> BlockShape,std::vector<double>& Result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}


	int TensorDims = TensorShape.size();
	int BlockDims = BlockShape.size();

	// create the buffers
	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, Tensor.size() * sizeof(double));
	cl::Buffer tensorShapeBuffer(context, CL_MEM_READ_ONLY, TensorShape.size() * sizeof(int));
	cl::Buffer blockStartBuffer(context, CL_MEM_READ_WRITE, BlockStart.size() * sizeof(int));
	cl::Buffer tensorDimsBuffer(context, CL_MEM_READ_ONLY, sizeof(int));
	
	
	cl::Buffer blockShapeBuffer(context, CL_MEM_READ_ONLY, BlockShape.size() * sizeof(int));
	cl::Buffer blockDimsBuffer(context, CL_MEM_READ_ONLY, sizeof(int));
	
	
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, Result.size() * sizeof(double));

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, Tensor.size() * sizeof(double), Tensor.data());
	queue.enqueueWriteBuffer(tensorShapeBuffer, CL_TRUE, 0, TensorShape.size() * sizeof(int), TensorShape.data());
	queue.enqueueWriteBuffer(blockStartBuffer, CL_TRUE, 0, BlockStart.size() * sizeof(int), BlockStart.data());
	queue.enqueueWriteBuffer(tensorDimsBuffer, CL_TRUE, 0, sizeof(int), &TensorDims);
	
	
	queue.enqueueWriteBuffer(blockShapeBuffer, CL_TRUE, 0, BlockShape.size() * sizeof(int), BlockShape.data());
	queue.enqueueWriteBuffer(blockDimsBuffer, CL_TRUE, 0, sizeof(int), &BlockDims);


	// set the arguments
	Tensor_Block_Kernel.setArg(0, tensorBuffer);
	Tensor_Block_Kernel.setArg(1, tensorShapeBuffer);
	Tensor_Block_Kernel.setArg(2, blockStartBuffer);
	Tensor_Block_Kernel.setArg(3, tensorDimsBuffer);


	Tensor_Block_Kernel.setArg(4, blockShapeBuffer);
	Tensor_Block_Kernel.setArg(5, blockDimsBuffer);

	Tensor_Block_Kernel.setArg(6, resultBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(Tensor_Block_Kernel, cl::NullRange, cl::NDRange(Result.size()), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, Result.size() * sizeof(double), Result.data());
}

static double randRange(double min, double max)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(min, max);
	return dis(gen);
}

class Tensor
{
	std::vector<int> shape;
	std::vector<double> data;

public:
	Tensor(std::vector<int> Shape);
	~Tensor();

	int getDimensions();
	std::vector<int> getShape();
	std::vector<double> getData();

	Tensor Block(std::vector<int> Start, std::vector<int> Shape);
	Tensor Transpose();
	Tensor Dot(Tensor &t);
	Tensor Conv(Tensor &Kernel, int Stride, int Padding);
	Tensor Mul(Tensor &t);
	double MulSum(Tensor &t);
	double Sum();
	void Randomize(double min, double max);
	void Print();
};