__global__ void kernel(int* i)
{
	i[threadIdx.x]++;
}

int main(int argc, char* argv[])
{
	kernel<<<1, 1>>>(reinterpret_cast<int*>(argv[0]));
	cudaDeviceSynchronize();

	return 0;
}

