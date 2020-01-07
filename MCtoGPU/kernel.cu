
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
using namespace std;
float timeGPU, timeCPU, factor;
#include "MCGPU.h"
#include "MCCPU.h"


int main(int argc,char* argv[])
{
	int wb,Count,start;
	
	if (argc > 1) {
		wb = atoi(argv[1]);
		if (argc > 2) {
			Count = atoi(argv[2]);
			if (argc > 3) {
				start = atoi(argv[3]);
			}
			else {
				start = 0;
			}
		}
		else {
			Count = 1;
			start = 0;
		}
	}
	else {
		cout << "-----------------------------------------------------" << endl;
		cout << "Ising model MC GPU/CPU Simulator by Aleksander Dawid" << endl;
		cout << "-----------------------------------------------------" << endl;
		cout << "Future usage: *.exe 64 20 18" << endl;
		cout << "64 - Threads per block, 20 - spin networks, 18 - starting network" << endl;
		cout << "-----------------------------------------------------" << endl;
		wb = 256;
		Count = 1;
		start = 0;
	}


	for (int test = start; test< Count; test++) {
		RunMCIsing(test,wb);
	}
}
