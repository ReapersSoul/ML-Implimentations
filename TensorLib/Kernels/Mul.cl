// Kernel function to multiply two vectors element by element
__kernel void vectorMul(__global const float* vectorA,
                        __global const float* vectorB,
                        __global float* result,
                        const int size)
{
    int i = get_global_id(0);
    
    if (i < size)
    {
        result[i] = vectorA[i] * vectorB[i];
    }
}
