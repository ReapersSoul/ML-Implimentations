__kernel void Sum_Step(__global const float* a, __global float* result) {
    int i = get_global_id(0);
	int index = i * 2;
    result[i] = a[index] + a[index + 1];
}
