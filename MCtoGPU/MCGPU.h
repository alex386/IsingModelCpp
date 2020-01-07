#pragma once

__constant__ int Nk;
__constant__ int Nj;
__constant__ int Ni;



class Vector {
public:
	int x, y, z;
	__host__ __device__ Vector() {
	}
};

__constant__ Vector wg[3];


int LoadToGPU(unsigned int N) {
	
	int *wsk = 0;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
		return 1;
	}


	cudaStatus = cudaMalloc((void**)&wsk, N * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		cerr << "Alokacja GPU niepoprawna" << endl;
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
	cout << "Alokacja GPU OK" << endl;
	getchar();
	return 0;
}

__device__ void ChangeSpinSpin(int idx, int *spin) {
	spin[idx] = -spin[idx];
}

__device__ void ChangeSpinSpin2(int idx) {
	extern __shared__ int lspin[];
	lspin[idx] = -lspin[idx];
}

__device__ int EnergyGPU(int idx, int *spin) {
	int E=0;
	int XX, YY, ZZ;
	int s, a, sp, p, q;
	int x, y, z;
	int Nkj = Nk * Nj;

	sp = spin[idx];
	
	Vector w[3];
	w[0].x = 1; w[0].y = 0; w[0].z = 0;
	w[1].x = 0; w[1].y = 1; w[1].z = 0;
	w[2].x = 0; w[2].y = 0; w[2].z = 1;

	z = idx / (Nkj);
	y = (idx - z * Nkj) / Nk;
	x = idx % Nk;

	for (p = 0; p < 2; p++) {
		s = 2 * p - 1;
		for (q = 0; q < 3; q++) {
			ZZ = z + s * w[q].x;
			if (ZZ < Nk && ZZ >= 0 ) {
				YY = y + s * w[q].y;
				if (YY < Nj && YY >= 0) {
					XX = x + s * w[q].z;
					if (XX < Ni && XX >= 0) {
						a = ZZ * Nkj + YY * Nj + XX;
						E += spin[a];
					}
				}
			}
		}
	}
	return -2 * sp*E;;
}

__global__ void ModelIsingaEven(int *spin) {
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	int Nkj = Nk * Nj;
	int a, b, idx;
	int Em, En;
	int i2;	

	i2 = 2 * i;	//Even
	b = (i2 / Nkj) % 2;
	a = ((i2 / Nk) % 2) * (1 - 2 * b);

	idx = i2 + a + b;      
	
	Em = EnergyGPU(idx,spin);
	ChangeSpinSpin(idx,spin);
	En = EnergyGPU(idx,spin);

	if (En>Em) {
		ChangeSpinSpin(idx, spin);
	}
}

__global__ void ModelIsingaOdd(int *spin) {
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int Nkj = Nk * Nj;
	int a, b, idx2;
	int Em, En;
	int i2;		
	i2 = 2 * i;
	b = (i2 / Nkj) % 2;
	a = ((i2 / Nk) % 2) * (1 - 2 * b);
	idx2 = i2 + 1 - a - b;   //Odd
	Em = EnergyGPU(idx2, spin);
	ChangeSpinSpin(idx2, spin);
	En = EnergyGPU(idx2, spin);
	if (En>Em) {
		ChangeSpinSpin(idx2, spin);
	}
}

