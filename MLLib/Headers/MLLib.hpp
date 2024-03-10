#pragma once
#include <stdexcept>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <filesystem>
#include <CL/cl2.hpp>

static double randRange(double min, double max)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(min, max);
	return dis(gen);
}

static std::vector<cl::Platform> platforms;
static std::vector<cl::Device> devices;
static cl::Context context;
static cl::CommandQueue queue;
static bool isInitialized = false;
static int maxWorkGroupSize;

// vector kernels and programs
static cl::Kernel Sum_Step_Kernel;
static cl::Program Sum_Step_Program;

static cl::Kernel Mul_Kernel;
static cl::Program Mul_Program;

static cl::Kernel Apply_Scalar_Kernel;
static cl::Program Apply_Scalar_Program;

static cl::Kernel Sub_Kernel;
static cl::Program Sub_Program;

static cl::Kernel Add_Kernel;
static cl::Program Add_Program;

static cl::Kernel Div_Kernel;
static cl::Program Div_Program;

// matrix kernels and programs
static cl::Kernel Matrix_Transpose_Kernel;
static cl::Program Matrix_Transpose_Program;

// tensor kernels and programs
static cl::Kernel Tensor_Block_Kernel;
static cl::Program Tensor_Block_Program;

static cl::Kernel Tensor_Transpose_Kernel;
static cl::Program Tensor_Transpose_Program;

static cl::Kernel Tensor_Dot_Kernel;
static cl::Program Tensor_Dot_Program;

static cl::Kernel Tensor_Conv_Kernel;
static cl::Program Tensor_Conv_Program;

//activation functions
static cl::Kernel Sigmoid_Kernel;
static cl::Program Sigmoid_Program;

static cl::Kernel ReLU_Kernel;
static cl::Program ReLU_Program;

static cl::Kernel LeakyReLU_Kernel;
static cl::Program LeakyReLU_Program;

static cl::Kernel Tanh_Kernel;
static cl::Program Tanh_Program;

static cl::Kernel Softmax_Kernel;
static cl::Program Softmax_Program;

static cl::Kernel GeLU_Kernel;
static cl::Program GeLU_Program;

//activation derivatives

static cl::Kernel Sigmoid_Derivative_Kernel;
static cl::Program Sigmoid_Derivative_Program;

static cl::Kernel ReLU_Derivative_Kernel;
static cl::Program ReLU_Derivative_Program;

static cl::Kernel LeakyReLU_Derivative_Kernel;
static cl::Program LeakyReLU_Derivative_Program;

static cl::Kernel Tanh_Derivative_Kernel;
static cl::Program Tanh_Derivative_Program;

static cl::Kernel Softmax_Derivative_Kernel;
static cl::Program Softmax_Derivative_Program;

static cl::Kernel GeLU_Derivative_Kernel;
static cl::Program GeLU_Derivative_Program;

//loss functions
static cl::Kernel MeanSquaredError_Kernel;
static cl::Program MeanSquaredError_Program;

static cl::Kernel MeanAbsoluteError_Kernel;
static cl::Program MeanAbsoluteError_Program;

static cl::Kernel BinaryCrossEntropy_Kernel;
static cl::Program BinaryCrossEntropy_Program;

static cl::Kernel CategoricalCrossEntropy_Kernel;
static cl::Program CategoricalCrossEntropy_Program;

//loss derivatives
static cl::Kernel MeanSquaredError_Derivative_Kernel;
static cl::Program MeanSquaredError_Derivative_Program;

static cl::Kernel MeanAbsoluteError_Derivative_Kernel;
static cl::Program MeanAbsoluteError_Derivative_Program;

static cl::Kernel BinaryCrossEntropy_Derivative_Kernel;
static cl::Program BinaryCrossEntropy_Derivative_Program;

static cl::Kernel CategoricalCrossEntropy_Derivative_Kernel;
static cl::Program CategoricalCrossEntropy_Derivative_Program;

//function to read the file into a string

static std::string slurp(std::ifstream &in)
{
	std::ostringstream sstr;
	sstr << in.rdbuf();
	return sstr.str();
}

//function to print the build error

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

		fprintf(stderr, "Build log for %s:\n%s\n", KernelName.c_str(), buff_erro); // Be careful with  the fprint
		free(buff_erro);
		fprintf(stderr, "clBuildProgram failed\n");
}

//function to initialize OpenCL and create the kernels and programs

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

	// vector kernels
	cl::Program::Sources sources;
	std::string kernel_code;
	std::string kernel_path = "../Kernels/Sum_Step.cl";
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

	kernel_path = "../Kernels/Mul.cl";
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

	kernel_path = "../Kernels/Apply_Scalar.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Apply_Scalar_Program = cl::Program(context, sources);
	err = Apply_Scalar_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Apply_Scalar",err,Apply_Scalar_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Apply_Scalar_Kernel = cl::Kernel(Apply_Scalar_Program, "Apply_Scalar");

	kernel_path = "../Kernels/Sub.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Sub_Program = cl::Program(context, sources);
	err = Sub_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Sub",err,Sub_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Sub_Kernel = cl::Kernel(Sub_Program, "Sub");

	kernel_path = "../Kernels/Add.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Add_Program = cl::Program(context, sources);
	err = Add_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Add",err,Add_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Add_Kernel = cl::Kernel(Add_Program, "Add");

	kernel_path = "../Kernels/Div.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Div_Program = cl::Program(context, sources);
	err = Div_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Div",err,Div_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Div_Kernel = cl::Kernel(Div_Program, "Div");

	// Matrix kernels
	kernel_path = "../Kernels/Matrix_Transpose.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Matrix_Transpose_Program = cl::Program(context, sources);
	err = Matrix_Transpose_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Matrix_Transpose",err,Matrix_Transpose_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Matrix_Transpose_Kernel = cl::Kernel(Matrix_Transpose_Program, "Matrix_Transpose");

	// Tensor kernels
	kernel_path = "../Kernels/Tensor_Block.cl";
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

	kernel_path = "../Kernels/Tensor_Transpose.cl";
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

	kernel_path = "../Kernels/Tensor_Dot.cl";
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

	kernel_path = "../Kernels/Tensor_Conv.cl";
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

	//activation functions
	kernel_path = "../Kernels/Activations/Sigmoid.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Sigmoid_Program = cl::Program(context, sources);
	err = Sigmoid_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Sigmoid",err,Sigmoid_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Sigmoid_Kernel = cl::Kernel(Sigmoid_Program, "Sigmoid");

	kernel_path = "../Kernels/Activations/ReLU.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	ReLU_Program = cl::Program(context, sources);
	err = ReLU_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("ReLU",err,ReLU_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	ReLU_Kernel = cl::Kernel(ReLU_Program, "ReLU");

	kernel_path = "../Kernels/Activations/LeakyReLU.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	LeakyReLU_Program = cl::Program(context, sources);
	err = LeakyReLU_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("LeakyReLU",err,LeakyReLU_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	LeakyReLU_Kernel = cl::Kernel(LeakyReLU_Program, "LeakyReLU");

	kernel_path = "../Kernels/Activations/Tanh.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Tanh_Program = cl::Program(context, sources);
	err = Tanh_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Tanh",err,Tanh_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Tanh_Kernel = cl::Kernel(Tanh_Program, "Tanh");

	kernel_path = "../Kernels/Activations/Softmax.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Softmax_Program = cl::Program(context, sources);
	err = Softmax_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Softmax",err,Softmax_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Softmax_Kernel = cl::Kernel(Softmax_Program, "Softmax");

	kernel_path = "../Kernels/Activations/GeLU.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	GeLU_Program = cl::Program(context, sources);
	err = GeLU_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("GeLU",err,GeLU_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	GeLU_Kernel = cl::Kernel(GeLU_Program, "GeLU");

	//activation derivatives
	kernel_path = "../Kernels/Activation_Derivatives/Sigmoid_Derivative.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Sigmoid_Derivative_Program = cl::Program(context, sources);
	err = Sigmoid_Derivative_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Sigmoid_Derivative",err,Sigmoid_Derivative_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Sigmoid_Derivative_Kernel = cl::Kernel(Sigmoid_Derivative_Program, "Sigmoid_Derivative");

	kernel_path = "../Kernels/Activation_Derivatives/ReLU_Derivative.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	ReLU_Derivative_Program = cl::Program(context, sources);
	err = ReLU_Derivative_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("ReLU_Derivative",err,ReLU_Derivative_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	ReLU_Derivative_Kernel = cl::Kernel(ReLU_Derivative_Program, "ReLU_Derivative");

	kernel_path = "../Kernels/Activation_Derivatives/LeakyReLU_Derivative.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	LeakyReLU_Derivative_Program = cl::Program(context, sources);
	err = LeakyReLU_Derivative_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("LeakyReLU_Derivative",err,LeakyReLU_Derivative_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	LeakyReLU_Derivative_Kernel = cl::Kernel(LeakyReLU_Derivative_Program, "LeakyReLU_Derivative");

	kernel_path = "../Kernels/Activation_Derivatives/Tanh_Derivative.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Tanh_Derivative_Program = cl::Program(context, sources);
	err = Tanh_Derivative_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Tanh_Derivative",err,Tanh_Derivative_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Tanh_Derivative_Kernel = cl::Kernel(Tanh_Derivative_Program, "Tanh_Derivative");

	kernel_path = "../Kernels/Activation_Derivatives/Softmax_Derivative.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	Softmax_Derivative_Program = cl::Program(context, sources);
	err = Softmax_Derivative_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("Softmax_Derivative",err,Softmax_Derivative_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	Softmax_Derivative_Kernel = cl::Kernel(Softmax_Derivative_Program, "Softmax_Derivative");

	kernel_path = "../Kernels/Activation_Derivatives/GeLU_Derivative.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	GeLU_Derivative_Program = cl::Program(context, sources);
	err = GeLU_Derivative_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("GeLU_Derivative",err,GeLU_Derivative_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	GeLU_Derivative_Kernel = cl::Kernel(GeLU_Derivative_Program, "GeLU_Derivative");

	//loss functions
	kernel_path = "../Kernels/Losses/MeanSquaredError.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	MeanSquaredError_Program = cl::Program(context, sources);
	err = MeanSquaredError_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("MeanSquaredError",err,MeanSquaredError_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	MeanSquaredError_Kernel = cl::Kernel(MeanSquaredError_Program, "MeanSquaredError");

	kernel_path = "../Kernels/Losses/MeanAbsoluteError.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	MeanAbsoluteError_Program = cl::Program(context, sources);
	err = MeanAbsoluteError_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("MeanAbsoluteError",err,MeanAbsoluteError_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	MeanAbsoluteError_Kernel = cl::Kernel(MeanAbsoluteError_Program, "MeanAbsoluteError");

	kernel_path = "../Kernels/Losses/BinaryCrossEntropy.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	BinaryCrossEntropy_Program = cl::Program(context, sources);
	err = BinaryCrossEntropy_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("BinaryCrossEntropy",err,BinaryCrossEntropy_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	BinaryCrossEntropy_Kernel = cl::Kernel(BinaryCrossEntropy_Program, "BinaryCrossEntropy");

	kernel_path = "../Kernels/Losses/CategoricalCrossEntropy.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	CategoricalCrossEntropy_Program = cl::Program(context, sources);
	err = CategoricalCrossEntropy_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("CategoricalCrossEntropy",err,CategoricalCrossEntropy_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	CategoricalCrossEntropy_Kernel = cl::Kernel(CategoricalCrossEntropy_Program, "CategoricalCrossEntropy");

	//loss derivatives
	kernel_path = "../Kernels/Loss_Derivatives/MeanSquaredError_Derivative.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	MeanSquaredError_Derivative_Program = cl::Program(context, sources);
	err = MeanSquaredError_Derivative_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("MeanSquaredError_Derivative",err,MeanSquaredError_Derivative_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	MeanSquaredError_Derivative_Kernel = cl::Kernel(MeanSquaredError_Derivative_Program, "MeanSquaredError_Derivative");
	
	kernel_path = "../Kernels/Loss_Derivatives/MeanAbsoluteError_Derivative.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	MeanAbsoluteError_Derivative_Program = cl::Program(context, sources);
	err = MeanAbsoluteError_Derivative_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("MeanAbsoluteError_Derivative",err,MeanAbsoluteError_Derivative_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	MeanAbsoluteError_Derivative_Kernel = cl::Kernel(MeanAbsoluteError_Derivative_Program, "MeanAbsoluteError_Derivative");

	kernel_path = "../Kernels/Loss_Derivatives/BinaryCrossEntropy_Derivative.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	BinaryCrossEntropy_Derivative_Program = cl::Program(context, sources);
	err = BinaryCrossEntropy_Derivative_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("BinaryCrossEntropy_Derivative",err,BinaryCrossEntropy_Derivative_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	BinaryCrossEntropy_Derivative_Kernel = cl::Kernel(BinaryCrossEntropy_Derivative_Program, "BinaryCrossEntropy_Derivative");

	kernel_path = "../Kernels/Loss_Derivatives/CategoricalCrossEntropy_Derivative.cl";
	// read whole file into string
	file.open(kernel_path);
	if (!file.is_open())
	{
		throw std::runtime_error("Could not open file: " + kernel_path);
	}
	kernel_code = slurp(file);
	file.close();
	sources.push_back({kernel_code.c_str(), kernel_code.length()});
	CategoricalCrossEntropy_Derivative_Program = cl::Program(context, sources);
	err = CategoricalCrossEntropy_Derivative_Program.build(devices);
	if (err != CL_SUCCESS)
	{
		PrintClBuildError("CategoricalCrossEntropy_Derivative",err,CategoricalCrossEntropy_Derivative_Program,devices[0],__LINE__);
		throw std::runtime_error("Could not build program: " + kernel_path);
	}
	CategoricalCrossEntropy_Derivative_Kernel = cl::Kernel(CategoricalCrossEntropy_Derivative_Program, "CategoricalCrossEntropy_Derivative");

	// set the flag
	isInitialized = true;
}

// vector functions
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
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = data.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(dataBuffer, CL_TRUE, 0, data.size() * sizeof(double), data.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Sum_Step_Kernel.setArg(0, dataBuffer);
	Sum_Step_Kernel.setArg(1, resultBuffer);
	Sum_Step_Kernel.setArg(2, sizeBuffer);

	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>(size/2)){
		maxWorkGroupSize = (size/2);
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Sum_Step_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

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
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	// write the data to the buffers
	queue.enqueueWriteBuffer(data1Buffer, CL_TRUE, 0, size * sizeof(double), data1.data());
	queue.enqueueWriteBuffer(data2Buffer, CL_TRUE, 0, size * sizeof(double), data2.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Mul_Kernel.setArg(0, data1Buffer);
	Mul_Kernel.setArg(1, data2Buffer);
	Mul_Kernel.setArg(2, resultBuffer);
	Mul_Kernel.setArg(3, sizeBuffer);
	
	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>size){
		maxWorkGroupSize = size;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Mul_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

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

static void GPU_Apply_Scalar(std::vector<double> tensor, double scalar, std::vector<double>&result){
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer scalarBuffer(context, CL_MEM_READ_ONLY, sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(scalarBuffer, CL_TRUE, 0, sizeof(double), &scalar);
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Mul_Kernel.setArg(0, tensorBuffer);
	Mul_Kernel.setArg(1, scalarBuffer);
	Mul_Kernel.setArg(2, resultBuffer);
	Mul_Kernel.setArg(3, sizeBuffer);

	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>size){
		maxWorkGroupSize = size;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Apply_Scalar_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_Sub(std::vector<double> tensor1, std::vector<double> tensor2, std::vector<double>&result){
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensor1Buffer(context, CL_MEM_READ_ONLY, tensor1.size() * sizeof(double));
	cl::Buffer tensor2Buffer(context, CL_MEM_READ_ONLY, tensor2.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor1.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor1.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensor1Buffer, CL_TRUE, 0, tensor1.size() * sizeof(double), tensor1.data());
	queue.enqueueWriteBuffer(tensor2Buffer, CL_TRUE, 0, tensor2.size() * sizeof(double), tensor2.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Sub_Kernel.setArg(0, tensor1Buffer);
	Sub_Kernel.setArg(1, tensor2Buffer);
	Sub_Kernel.setArg(2, resultBuffer);
	Sub_Kernel.setArg(3, sizeBuffer);

	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>size){
		maxWorkGroupSize = size;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Sub_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_Add(std::vector<double> tensor1, std::vector<double> tensor2, std::vector<double>&result){
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensor1Buffer(context, CL_MEM_READ_ONLY, tensor1.size() * sizeof(double));
	cl::Buffer tensor2Buffer(context, CL_MEM_READ_ONLY, tensor2.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor1.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor1.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensor1Buffer, CL_TRUE, 0, tensor1.size() * sizeof(double), tensor1.data());
	queue.enqueueWriteBuffer(tensor2Buffer, CL_TRUE, 0, tensor2.size() * sizeof(double), tensor2.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Add_Kernel.setArg(0, tensor1Buffer);
	Add_Kernel.setArg(1, tensor2Buffer);
	Add_Kernel.setArg(2, resultBuffer);
	Add_Kernel.setArg(3, sizeBuffer);

	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>size){
		maxWorkGroupSize = size;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Add_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_Div(std::vector<double> tensor1, std::vector<double> tensor2, std::vector<double>&result){
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensor1Buffer(context, CL_MEM_READ_ONLY, tensor1.size() * sizeof(double));
	cl::Buffer tensor2Buffer(context, CL_MEM_READ_ONLY, tensor2.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor1.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor1.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensor1Buffer, CL_TRUE, 0, tensor1.size() * sizeof(double), tensor1.data());
	queue.enqueueWriteBuffer(tensor2Buffer, CL_TRUE, 0, tensor2.size() * sizeof(double), tensor2.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Div_Kernel.setArg(0, tensor1Buffer);
	Div_Kernel.setArg(1, tensor2Buffer);
	Div_Kernel.setArg(2, resultBuffer);
	Div_Kernel.setArg(3, sizeBuffer);

	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>size){
		maxWorkGroupSize = size;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Div_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

// Matrix functions
static void GPU_TransposeMatrix(std::vector<std::vector<double>> Matrix, std::vector<std::vector<double>> &Result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	int Rows = Matrix.size();
	int Cols = Matrix[0].size();

	// create the buffers
	cl::Buffer matrixBuffer(context, CL_MEM_READ_ONLY, Rows * Cols * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, Rows * Cols * sizeof(double));
	cl::Buffer rowsBuffer(context, CL_MEM_READ_ONLY, sizeof(int));
	cl::Buffer colsBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	// write the data to the buffers
	queue.enqueueWriteBuffer(matrixBuffer, CL_TRUE, 0, Rows * Cols * sizeof(double), Matrix.data());
	queue.enqueueWriteBuffer(rowsBuffer, CL_TRUE, 0, sizeof(int), &Rows);
	queue.enqueueWriteBuffer(colsBuffer, CL_TRUE, 0, sizeof(int), &Cols);

	// set the arguments
	Matrix_Transpose_Kernel.setArg(0, matrixBuffer);
	Matrix_Transpose_Kernel.setArg(1, resultBuffer);
	Matrix_Transpose_Kernel.setArg(2, rowsBuffer);
	Matrix_Transpose_Kernel.setArg(3, colsBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(Matrix_Transpose_Kernel, cl::NullRange, cl::NDRange(Rows, Cols), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, Rows * Cols * sizeof(double), Result.data());
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

// Activation functions
static void GPU_Sigmoid(std::vector<double> tensor, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Sigmoid_Kernel.setArg(0, tensorBuffer);
	Sigmoid_Kernel.setArg(1, resultBuffer);
	Sigmoid_Kernel.setArg(2, sizeBuffer);

	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>size){
		maxWorkGroupSize = size;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Sigmoid_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_ReLU(std::vector<double> tensor, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	ReLU_Kernel.setArg(0, tensorBuffer);
	ReLU_Kernel.setArg(1, resultBuffer);
	ReLU_Kernel.setArg(2, sizeBuffer);

	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>size){
		maxWorkGroupSize = size;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(ReLU_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_LeakyReLU(std::vector<double> tensor, std::vector<double> &result, double alpha)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));
	cl::Buffer alphaBuffer(context, CL_MEM_READ_ONLY, sizeof(double));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);
	queue.enqueueWriteBuffer(alphaBuffer, CL_TRUE, 0, sizeof(double), &alpha);

	// set the arguments
	LeakyReLU_Kernel.setArg(0, tensorBuffer);
	LeakyReLU_Kernel.setArg(1, resultBuffer);
	LeakyReLU_Kernel.setArg(2, sizeBuffer);
	LeakyReLU_Kernel.setArg(3, alphaBuffer);

	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>size){
		maxWorkGroupSize = size;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(LeakyReLU_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_Tanh(std::vector<double> tensor, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Tanh_Kernel.setArg(0, tensorBuffer);
	Tanh_Kernel.setArg(1, resultBuffer);
	Tanh_Kernel.setArg(2, sizeBuffer);

	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>size){
		maxWorkGroupSize = size;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Tanh_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_Softmax(std::vector<double> tensor, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Softmax_Kernel.setArg(0, tensorBuffer);
	Softmax_Kernel.setArg(1, resultBuffer);
	Softmax_Kernel.setArg(2, sizeBuffer);

	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>size){
		maxWorkGroupSize = size;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(Softmax_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_GeLU(std::vector<double> tensor, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	GeLU_Kernel.setArg(0, tensorBuffer);
	GeLU_Kernel.setArg(1, resultBuffer);
	GeLU_Kernel.setArg(2, sizeBuffer);

	//get the max work group size
	size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if(maxWorkGroupSize>size){
		maxWorkGroupSize = size;
	}

	// execute the kernel
	queue.enqueueNDRangeKernel(GeLU_Kernel, cl::NullRange, cl::NDRange(maxWorkGroupSize), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

// Activation derivatives
static void GPU_Sigmoid_Derivative(std::vector<double> tensor, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Sigmoid_Derivative_Kernel.setArg(0, tensorBuffer);
	Sigmoid_Derivative_Kernel.setArg(1, resultBuffer);
	Sigmoid_Derivative_Kernel.setArg(2, sizeBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(Sigmoid_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_ReLU_Derivative(std::vector<double> tensor, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	ReLU_Derivative_Kernel.setArg(0, tensorBuffer);
	ReLU_Derivative_Kernel.setArg(1, resultBuffer);
	ReLU_Derivative_Kernel.setArg(2, sizeBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(ReLU_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_LeakyReLU_Derivative(std::vector<double> tensor, std::vector<double> &result, double alpha)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));
	cl::Buffer alphaBuffer(context, CL_MEM_READ_ONLY, sizeof(double));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);
	queue.enqueueWriteBuffer(alphaBuffer, CL_TRUE, 0, sizeof(double), &alpha);

	// set the arguments
	LeakyReLU_Derivative_Kernel.setArg(0, tensorBuffer);
	LeakyReLU_Derivative_Kernel.setArg(1, resultBuffer);
	LeakyReLU_Derivative_Kernel.setArg(2, sizeBuffer);
	LeakyReLU_Derivative_Kernel.setArg(3, alphaBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(LeakyReLU_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_Tanh_Derivative(std::vector<double> tensor, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Tanh_Derivative_Kernel.setArg(0, tensorBuffer);
	Tanh_Derivative_Kernel.setArg(1, resultBuffer);
	Tanh_Derivative_Kernel.setArg(2, sizeBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(Tanh_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_Softmax_Derivative(std::vector<double> tensor, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	Softmax_Derivative_Kernel.setArg(0, tensorBuffer);
	Softmax_Derivative_Kernel.setArg(1, resultBuffer);
	Softmax_Derivative_Kernel.setArg(2, sizeBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(Softmax_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

static void GPU_GeLU_Derivative(std::vector<double> tensor, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	cl::Buffer tensorBuffer(context, CL_MEM_READ_ONLY, tensor.size() * sizeof(double));
	cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, tensor.size() * sizeof(double));
	cl::Buffer sizeBuffer(context, CL_MEM_READ_ONLY, sizeof(int));

	int size = tensor.size();

	// write the data to the buffers
	queue.enqueueWriteBuffer(tensorBuffer, CL_TRUE, 0, tensor.size() * sizeof(double), tensor.data());
	queue.enqueueWriteBuffer(sizeBuffer, CL_TRUE, 0, sizeof(int), &size);

	// set the arguments
	GeLU_Derivative_Kernel.setArg(0, tensorBuffer);
	GeLU_Derivative_Kernel.setArg(1, resultBuffer);
	GeLU_Derivative_Kernel.setArg(2, sizeBuffer);

	// execute the kernel
	queue.enqueueNDRangeKernel(GeLU_Derivative_Kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

	// read the result
	queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, size * sizeof(double), result.data());
}

// Loss functions
static void GPU_MeanSquaredError(std::vector<double> output, std::vector<double> target, double &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	std::vector<double> temp;
	temp.resize(output.size());
	GPU_Sub(output, target, temp);
	GPU_Mul(temp, temp, temp);
	GPU_Sum(temp, result);
	result = result / output.size();
}

static void GPU_MeanAbsoluteError(std::vector<double> output, std::vector<double> target, double &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	std::vector<double> temp;
	temp.resize(output.size());
	GPU_Sub(output, target, temp);
	for (int i = 0; i < temp.size(); i++)
	{
		if (temp[i] < 0)
		{
			temp[i] = -temp[i];
		}
	}
	GPU_Sum(temp, result);
	result = result / output.size();
}

static void GPU_LogLoss(std::vector<double> output, std::vector<double> target, double &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	std::vector<double> temp;
	temp.resize(output.size());
	GPU_Apply_Scalar(output, -1, temp);
	GPU_Apply_Scalar(temp, 1, temp);
	GPU_Mul(target, temp, temp);
	GPU_Sum(temp, result);
	result = result / output.size();
}

static void GPU_CrossEntropy(std::vector<double> output, std::vector<double> target, double &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	std::vector<double> temp;
	temp.resize(output.size());
	GPU_Apply_Scalar(output, -1, temp);
	GPU_Apply_Scalar(temp, 1, temp);
	GPU_Mul(target, temp, temp);
	GPU_Sum(temp, result);
	result = result / output.size();
}

// Loss derivatives
static void GPU_MeanSquaredError_Derivative(std::vector<double> output, std::vector<double> target, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	GPU_Sub(output, target, result);
	GPU_Apply_Scalar(result, 2.0 / output.size(), result);
}

static void GPU_MeanAbsoluteError_Derivative(std::vector<double> output, std::vector<double> target, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	GPU_Sub(output, target, result);
	for (int i = 0; i < result.size(); i++)
	{
		if (result[i] < 0)
		{
			result[i] = -1;
		}
		else
		{
			result[i] = 1;
		}
	}
	GPU_Apply_Scalar(result, 1.0 / output.size(), result);
}

static void GPU_LogLoss_Derivative(std::vector<double> output, std::vector<double> target, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	std::vector<double> temp;
	temp.resize(output.size());
	GPU_Apply_Scalar(output, -1, temp);
	GPU_Apply_Scalar(temp, 1, temp);
	GPU_Div(target, temp, result);
	GPU_Apply_Scalar(result, 1.0 / output.size(), result);
}

static void GPU_CrossEntropy_Derivative(std::vector<double> output, std::vector<double> target, std::vector<double> &result)
{
	// check if OpenCL is initialized
	if (!isInitialized)
	{
		InitializeOpenCL();
	}

	std::vector<double> temp;
	temp.resize(output.size());
	GPU_Apply_Scalar(output, -1, temp);
	GPU_Apply_Scalar(temp, 1, temp);
	GPU_Div(target, temp, result);
	GPU_Apply_Scalar(result, 1.0 / output.size(), result);
}

// Tensor class
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
	void MakeIndexTensor();
	void Print();
};

// Activation functions
class ActivationFunction
{
public:
	virtual double Activate(double input) = 0;
	virtual double Derivative(double input) = 0;
	virtual std::vector<double> Activate(std::vector<double> input) = 0;
	virtual std::vector<double> Derivative(std::vector<double> input) = 0;
};

class Sigmoid : public ActivationFunction
{
public:
	double Activate(double input);
	double Derivative(double input);
	std::vector<double> Activate(std::vector<double> input);
	std::vector<double> Derivative(std::vector<double> input);
};

class ReLU : public ActivationFunction
{
public:
	double Activate(double input);
	double Derivative(double input);
	std::vector<double> Activate(std::vector<double> input);
	std::vector<double> Derivative(std::vector<double> input);
};

class LeakyReLU : public ActivationFunction
{
public:
	double alpha;
	LeakyReLU(double Alpha);
	double Activate(double input);
	double Derivative(double input);
	std::vector<double> Activate(std::vector<double> input);
	std::vector<double> Derivative(std::vector<double> input);
};

class Tanh : public ActivationFunction
{
public:
	double Activate(double input);
	double Derivative(double input);
	std::vector<double> Activate(std::vector<double> input);
	std::vector<double> Derivative(std::vector<double> input);
};

class Softmax : public ActivationFunction
{
public:
	double Activate(double input);
	double Derivative(double input);
	std::vector<double> Activate(std::vector<double> input);
	std::vector<double> Derivative(std::vector<double> input);
};

class GeLU : public ActivationFunction
{
public:
	double Activate(double input);
	double Derivative(double input);
	std::vector<double> Activate(std::vector<double> input);
	std::vector<double> Derivative(std::vector<double> input);
};

// Loss functions
class LossFunction
{
public:
	virtual double Calculate(double output, double target) = 0;
	virtual double Derivative(double output, double target) = 0;
	virtual std::vector<double> Calculate(std::vector<double> output, std::vector<double> target) = 0;
	virtual std::vector<double> Derivative(std::vector<double> output, std::vector<double> target) = 0;
};

class MeanSquaredError : public LossFunction
{
public:
	double Calculate(double output, double target);
	double Derivative(double output, double target);
	std::vector<double> Calculate(std::vector<double> output, std::vector<double> target);
	std::vector<double> Derivative(std::vector<double> output, std::vector<double> target);
};

class MeanAbsoluteError : public LossFunction
{
public:
	double Calculate(double output, double target);
	double Derivative(double output, double target);
	std::vector<double> Calculate(std::vector<double> output, std::vector<double> target);
	std::vector<double> Derivative(std::vector<double> output, std::vector<double> target);
};

class LogLoss : public LossFunction
{
public:
	double Calculate(double output, double target);
	double Derivative(double output, double target);
	std::vector<double> Calculate(std::vector<double> output, std::vector<double> target);
	std::vector<double> Derivative(std::vector<double> output, std::vector<double> target);
};

class CrossEntropy : public LossFunction
{
public:
	double Calculate(double output, double target);
	double Derivative(double output, double target);
	std::vector<double> Calculate(std::vector<double> output, std::vector<double> target);
	std::vector<double> Derivative(std::vector<double> output, std::vector<double> target);
};


