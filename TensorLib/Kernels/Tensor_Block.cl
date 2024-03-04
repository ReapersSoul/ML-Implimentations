int convertToFlatIndex(int *dimensions, int *position, int dimensions_size) {
  // initialize the flat index
  int flatIndex = 0;

  // calculate the flat index
  for (int i = 0; i < dimensions_size; i++) {
    // calculate the flat index
    flatIndex = flatIndex * dimensions[i] + position[i];
  }

  return flatIndex;
}

int GetLastIndex(int *Shape, int size) {
  return convertToFlatIndex(Shape, Shape, size);
}

int *zeros(int *shape, int dims) {
  int size = GetLastIndex(shape, dims);
  int *result = new int[size];
  for (int i = 0; i < size; i++) {
    result[i] = 0;
  }
  return result;
}

// move from 3,4,5 to 3,4,4 to 3,4,3 and so on until 0,0,0
int *decrement(int *position, int *shape, int dims) {
  // if the position cannot be decremented, return the position
  bool canDecrement = false;
  for (int i = 0; i < dims; i++) {
    if (position[i] > 0) {
      canDecrement = true;
      break;
    }
  }
  if (!canDecrement) {
    return position;
  }

  for (int i = dims - 1; i >= 0; i--) {
    if (position[i] > 0) {
      position[i]--;
      return position;
    } else {
      position[i] = shape[i] - 1;
    }
  }
  return position;
}

// move from 0,0,0 to 0,0,1 to 0,0,2 and so on until 3,4,5
int *increment(int *position, int *shape, int dims) {
  // if the position cannot be incremented, return the position
  bool canIncrement = false;
  for (int i = 0; i < dims; i++) {
    if (position[i] < shape[i] - 1) {
      canIncrement = true;
      break;
    }
  }
  if (!canIncrement) {
    return position;
  }

  for (int i = dims - 1; i >= 0; i--) {
    if (position[i] < shape[i] - 1) {
      position[i]++;
      return position;
    } else {
      position[i] = 0;
    }
  }
  return position;
}

int *incrementBy(int *position, int *shape, int dims, int amount) {
  for (int i = 0; i < amount; i++) {
    position = increment(position, shape, dims);
  }
  return position;
}

int *add_elements(int *position1, int *position2, int *shape, int dims) {
  int *result = zeros(shape, dims);
  for (int i = 0; i < dims; i++) {
    result[i] = position1[i] + position2[i];
  }
  return result;
}

__kernel void Tensor_Block(__global double *tensor,
                           __global double *tensor_shape,
                           __global int tensor_dims, __global int *start,
                           __global int *block_shape, __global int block_dim,
                           __global double *result) {
  // Get the global ID
  int gid = get_global_id(0);
  int *zero_position = zeros(block_shape, block_dim);
  int *block_position = incrementBy(zero_position, block_shape, block_dim, gid);
  int TensorIndex = convertToFlatIndex(
      tensor_shape, add_elements(start, block_position, tensor_dims),
      tensor_dims);
  int BlockIndex = convertToFlatIndex(block_shape, block_position, block_dim);
  result[BlockIndex] = tensor[TensorIndex];
}
