#pragma once
/*====================================================*/
/* Symulacja Monte Carlo spinów 1D,2D,3D Isinga       */
/* Sieæ SC                                            */
/* Autor: Aleksander Dawid                            */
/*====================================================*/
#include <algorithm>
#include <time.h>
#include <fstream>
#include <Windows.h>

class IsingModel {
	int x, y, z;
	Vector w[3];
public:
	int Nx, Ny, Nz;
	int N;
	int BlockPerGrid;               
	int ThreadPerBlock;               
	int *Spin;                    
	int *SpinNetwork, *SpinTest;
	IsingModel(int, int, int);
	~IsingModel();
	int* GPUAllocation(unsigned int N);
	int GPU(int tN);
	void CPU(int tN);
	void ReadTempData();
	int WriteSpinToGPU();
	void EvenSpin(int i);
	void OddSpin(int i);
	void ChangeSpin();
	int Energy();
	int TotalEnergy();
};

int IsingModel::GPU(int tN) {
	cudaError_t cudaStatus;

	float timer;
	cudaEvent_t start, stop;
	
	
	BlockPerGrid = (N / 2) / ThreadPerBlock;
	if ((Nx / 8) % 2 != 0) { BlockPerGrid++; }
	cout << "SC: " << "(" << Nx << "," << Ny << "," << Nz << ")" << endl;
	cout << "Time: " << tN << endl;
	cout << "N/2: " << N/2 << endl;
	cout << "ThreadPerBlock: " << ThreadPerBlock << endl;
	cout << "BlockPerGrid: " << BlockPerGrid << endl;
	cout << "Block*Thread: " << BlockPerGrid* ThreadPerBlock << endl;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
   for(int t=0;t<tN;t++){
	ModelIsingaEven <<<BlockPerGrid, ThreadPerBlock>>> (Spin);
	//Global synchronization
	ModelIsingaOdd << <BlockPerGrid, ThreadPerBlock >> > (Spin);
   }

   cudaStatus = cudaGetLastError();
   if (cudaStatus != cudaSuccess) {
	   fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	   return 1;
   }

   cudaStatus = cudaDeviceSynchronize();
   if (cudaStatus != cudaSuccess) {
	   fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	   return 1;
   }
   
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer, start, stop);
	timeGPU = timer / 1000.0f;
	cout << "GPU time: " << timeGPU << " s" << endl;

	cudaStatus = cudaMemcpy(SpinNetwork, Spin, N * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMemcpy failed!" << endl;
	}

	return 0;
}

int* IsingModel::GPUAllocation(unsigned int N) {
	int *wsk=0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cerr<<"No CUDA enabled device in the system!"<<endl;
		return NULL;
	}
	cudaStatus = cudaMalloc((void**)&wsk, N*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		cerr << "GPU allocation failed" << endl;
		return NULL;
	}
	return wsk;
}

int IsingModel::TotalEnergy() {
	int E = 0;
	for (z = 0; z < Nz; z++)
	{
		for (y = 0; y < Ny; y++)
		{
			for (x = 0; x < Nx; x++)
			{
				E += Energy();
			}
		}
	}
	return E;
}


int IsingModel::Energy() {
	int E = 0;
	int s,a;
	int XX, YY, ZZ;
	int Nzy = Nz * Ny;
	int nr = z * Nzy + y * Nx + x;
	int nrSpin = SpinNetwork[nr];
	
	for (int p = 0; p < 2; p++) {
		s = 2 * p - 1;
		for (int q = 0; q < 3; q++) {
			ZZ = z + s * w[q].x;
			if (ZZ < Nz && ZZ >= 0) {
				YY = y + s * w[q].y;
				if (YY < Ny && YY >= 0) {
					XX = x + s * w[q].z;
					if (XX < Nx && XX >= 0) {
						a = ZZ * Nzy + YY * Ny + XX;
						
						E += SpinNetwork[a];
					}
				}
			}
		}
	}
	return -2* nrSpin*E;
}

void IsingModel::ChangeSpin() {
	int nr = z * Nz*Ny + y * Nx + x;
	SpinNetwork[nr] = -SpinNetwork[nr];
}


void IsingModel::EvenSpin(int i) {
	int a, b;
	int pp;

		b = (2 * i / (Nz*Ny)) % 2;
		a = ((2 * i / Nz) % 2) * (1 - 2 * b);
		pp = 2 * i + a + b;       //Even

		z = pp / (Nz*Ny);
		y = (pp - z * Nz*Ny) / Nz;
		x = pp % Nz;		
}

void IsingModel::OddSpin(int i) {
	int a, b;
	int np;

	b = (2 * i / (Nz*Ny)) % 2;
	a = ((2 * i / Nz) % 2) * (1 - 2 * b);
	np = 2 * i + 1 - (a + b);   //Odd

	z = np / (Nz*Ny);
	y = (np - z * Nz*Ny) / Nz;
	x = np % Nz;
}


void IsingModel::ReadTempData(){
	for(int i=0;i<N;i++){
	 SpinNetwork[i]=SpinTest[i];
	}
}

IsingModel::IsingModel(int Nx, int Ny, int Nz) {	
	int nr;
	cudaMemcpyToSymbol(Nk, &Nz, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Nj, &Ny, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Ni, &Nx, sizeof(int), 0, cudaMemcpyHostToDevice);

	this->Nx = Nx;
	this->Ny = Ny;
	this->Nz = Nz;
	N = Nx*Ny*Nz;
	Spin = GPUAllocation(N);
	SpinNetwork = new int[N];
	SpinTest = new int[N];
	srand((unsigned int)time(NULL));
	for (int z = 0; z < Nz; z++)
	{
		for (int y = 0; y < Ny; y++)
		{
			for (int x = 0; x < Nx; x++)
			{
				nr = z * Nz*Ny + y * Nx + x;
				SpinNetwork[nr] = 2 * (rand() % 2) - 1;
				SpinTest[nr] = SpinNetwork[nr];
			}
		}
	}
	
	WriteSpinToGPU();

	w[0].x = 1; w[0].y = 0; w[0].z = 0;
	w[1].x = 0; w[1].y = 1; w[1].z = 0;
	w[2].x = 0; w[2].y = 0; w[2].z = 1;

	cudaMemcpyToSymbol(wg, &w, 3*sizeof(Vector), 0, cudaMemcpyHostToDevice);
}

void IsingModel::CPU(int tN) {
	int Em, En, eN;
	eN = N / 2;
	cout << "#spins: " << eN*2 << endl;
	for (int t = 0; t < tN; t++) {
		//Even 2n
		for (int i = 0; i < eN; i++) {
			EvenSpin(i);
			Em = Energy();
			ChangeSpin();
			En = Energy();
			if (En > Em) {
				ChangeSpin();
			}
		}
		//Odd 2n+1
		for (int i = 0; i < eN; i++) {
			OddSpin(i);
			Em = Energy();
			ChangeSpin();
			En = Energy();
			if (En > Em) {
				ChangeSpin();
			}
		}
	}
}

int IsingModel::WriteSpinToGPU() {
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(Spin, SpinTest, N * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cerr << "Copy failed !" << endl;
		return 1;
	}
	return 0;
}

IsingModel::~IsingModel() {
	cudaError_t cudaStatus;
	delete[] SpinNetwork;
	delete[] SpinTest;
	cudaStatus=cudaFree(Spin);
	if (cudaStatus != cudaSuccess) {
		cerr << "GPU memory problem!" << endl;
	}
}

float TimeDiff()
{
	// Last counter reading
	static LARGE_INTEGER OldCounter = { 0, 0 };

	LARGE_INTEGER Counter, Frequency;
	if (QueryPerformanceFrequency(&Frequency))
	{
		// Gets current counter reading
		QueryPerformanceCounter(&Counter);

		// Calculates time difference (zero if called first time)
		float TimeDiff = OldCounter.LowPart ? (float)(Counter.LowPart - OldCounter.LowPart) / Frequency.LowPart : 0;

		// Resets last counter reading
		OldCounter = Counter;

		// Returns time difference
		return TimeDiff;
	}
	else
	{
		// No high resolution performance counter; returns zero
		return 0;
	}
}

void RunMCIsing(int grid,int wb) {
	int tN;
	tN = 1000;
	
	cout << "tN=" << tN << endl;
	int gr = 8 +grid*8;
	int E_CPU, E_GPU;

	IsingModel MC(gr, gr, gr);
	
		//MC.ReadTempData();
		cout << "----------------GPU-------------------" << endl;
		cout << "Starting energy: " << MC.TotalEnergy() << endl;
		MC.ThreadPerBlock = wb;
		MC.GPU(tN);
		E_GPU = MC.TotalEnergy();
		cout << "Final energy: " << E_GPU << endl;
		MC.ReadTempData();
		cout << "----------------CPU-------------------" << endl;
		cout << "Starting energy: " << MC.TotalEnergy() << endl;
		float dif;
		TimeDiff();
		MC.CPU(tN);
		dif = TimeDiff();
		cout << "CPU time: " << dif << " s " << endl;
		timeCPU = dif;
		E_CPU = MC.TotalEnergy();
		cout << "Final energy: " << E_CPU << endl;


		factor = timeCPU / timeGPU;
		cout << "----------------SUMMARY-----------------" << endl;
		cout << "Speedup: " << factor << " times" << endl;
		cout << "Energy difference E_CPU-E_GPU: " << E_CPU - E_GPU << endl;

		char FileN[256];
		sprintf(FileN, "Data%d.txt", MC.ThreadPerBlock);
		ofstream dopliku(FileN, ios::out | ios::app);
		dopliku << MC.N << "\t" << factor << endl;
		dopliku.close();
}
