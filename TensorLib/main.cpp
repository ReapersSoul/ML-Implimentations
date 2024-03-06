#include "TensorLib.hpp"

int position_to_flat_index(const std::vector<int>& position, const std::vector<int>& shape){
    int index = 0;
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        index += position[i] * stride;
        stride *= shape[i];
    }
    return index;
}


void IncrementPosition(std::vector<int>& position, const std::vector<int>& shape, int dims, int advance_by) {
    for (int i = dims - 1; i >= 0 && advance_by > 0; --i) {
        int remaining_space = shape[i] - position[i] - 1;
        int to_add = std::min(advance_by, remaining_space);
        position[i] += to_add;
        advance_by -= to_add;
        if (advance_by > 0)
            position[i] = 0; // reset the current dimension to 0 and carry over
    }
}

void DecrementPosition(std::vector<int>& position, const std::vector<int>& shape, int dims, int decrement_by) {
    for (int i = dims - 1; i >= 0 && decrement_by > 0; --i) {
        int to_subtract = std::min(decrement_by, position[i]);
        position[i] -= to_subtract;
        decrement_by -= to_subtract;
        if (decrement_by > 0 && i > 0) {
            position[i] = shape[i] - 1; // set the current dimension to the maximum and carry over
        }
    }
}
template <typename T>
void print_vector(std::vector<T> v){
	for (int i = 0; i < v.size(); i++){
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}

int posToIndex(int* pos, int* size, int dims) {
	int index = 0;
	int stride = 1;
	for (int i = 0; i < dims; i++) {
		index += pos[i] * stride;
		if (i < dims - 1) {
			stride *= size[i];
		}
	}
	return index;
}

int indexToPos(int index, int* size, int dim, int dims) {
    int stride = 1;
    for (int i = 0; i < dim; i++) {
        stride *= size[i];
    }
    return (index / stride) % size[dim];
}

int GetIndexOfBlockWithinTensorWithOffset(int* TensorShape,int* BlockStart,  int* BlockShape, int TensorDims, int BlockDims,int offset){
  int index = 0;
  int multiplier = 1;

  for(int i = 0; i < TensorDims; i++){
    index += (BlockStart[i] + indexToPos(offset, BlockShape, i, BlockDims)) * multiplier;
    multiplier *= TensorShape[i];
  }
  
  return index;
}

void TTensor_Block(double* Tensor,int* TensorShape,int* BlockStart,int * TensorDims,int* BlockShape,int* BlockDims,double* Result)
{
    for(int gid = 0; gid < BlockShape[0]*BlockShape[1]*BlockShape[2]; gid++){
        Result[gid] = GetIndexOfBlockWithinTensorWithOffset(TensorShape, BlockStart, BlockShape, *TensorDims, *BlockDims, gid);
    }
}

int main()
{
    Tensor t1({5, 5, 5});
    t1.Randomize(-1,1);
    Tensor t2=t1.Block({0, 0, 0}, {2, 2, 2});
    t1.Print();
    t2.Print();

	return 0;
}