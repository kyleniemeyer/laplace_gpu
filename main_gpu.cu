#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cutil.h>

#include <time.h>
//#include "walltime.h"

#define Real double

typedef unsigned int uint;

/** SOR relaxation parameter */
const Real omega = 1.85;

// block size
const uint block_size = 64;

void fill_coeffs (uint rowmax, uint colmax, Real th_cond, Real dx, Real dy,
				  				Real width, Real TN, Real * aP, Real * aW, Real * aE, 
				  				Real * aS, Real * aN, Real * b)
{  
	for (uint col = 0; col < colmax; ++col) {
		for (uint row = 0; row < rowmax; ++row) {
			uint ind = col * rowmax + row;
			
			b[ind] = 0.0;
			Real SP = 0.0;
			
			if (col == 0) {
				// left BC: temp = 0
				aW[ind] = 0.0;
				SP = -2.0 * th_cond * width * dy / dx;
			} else {
				aW[ind] = th_cond * width * dy / dx;
			}
			
			if (col == colmax - 1) {
				// right BC: temp = 0
				aE[ind] = 0.0;
				SP = -2.0 * th_cond * width * dy / dx;
			} else {
				aE[ind] = th_cond * width * dy / dx;
			}
			
			if (row == 0) {
				// bottom BC: temp = 0
				aS[ind] = 0.0;
				SP = -2.0 * th_cond * width * dx / dy;
			} else {
				aS[ind] = th_cond * width * dx / dy;
			}
			
			if (row == rowmax - 1) {
				// top BC: temp = TN
				aN[ind] = 0.0;
				b[ind] = 2.0 * th_cond * width * dx * TN / dy;
				SP = -2.0 * th_cond * width * dx / dy;
			} else {
				aN[ind] = th_cond * width * dx / dy;
			}
			
			aP[ind] = aW[ind] + aE[ind] + aS[ind] + aN[ind] - SP;
		} // end for row
	} // end for col
}

__global__ void red_kernel (uint rowmax, uint colmax, const Real * aP,
														const Real * aW, const Real * aE, const Real * aS,
														const Real * aN, const Real * b, Real * temp, Real * bl_norm_L2)
{	
	uint row = 1 + (blockIdx.y * blockDim.y) + threadIdx.y;
	uint col = 1 + (blockIdx.x * blockDim.x) + threadIdx.x;

	// store residual for block
	__shared__ Real res_cache[block_size];
	
	res_cache[threadIdx.y] = 0.0;
		
	// only red cell if even
	if ((row + col) % 2 == 0) {
		uint ind = (col - 1) * (rowmax - 2) + (row - 1);
		uint ind2 = col * rowmax + row;
	
		Real res = b[ind] + (aW[ind] * temp[(col - 1) * rowmax + row]
										   + aE[ind] * temp[(col + 1) * rowmax + row]
										   + aS[ind] * temp[col * rowmax + (row - 1)]
										   + aN[ind] * temp[col * rowmax + (row + 1)]);
	
		//temp[ind2] = temp[ind2] * (1.0 - omega) + omega * (res / aP[ind]);
		Real temp_old = temp[ind2];
		Real temp_new = temp_old * (1.0 - omega) + omega * (res / aP[ind]);
		
		temp[ind2] = temp_new;
		res = temp_old - temp_new;
		
		// store squared residual from each thread
		res_cache[threadIdx.y] = res * res;
		
		// synchronize threads in block
		__syncthreads();
		
		// add up squared residuals for block
		uint i = block_size / 2;
		while (i != 0) {
			if (threadIdx.y < i) {
				res_cache[threadIdx.y] += res_cache[threadIdx.y + i];
			}
			__syncthreads();
			i /= 2;
		}
		
		// store block's summed residuals
		if (threadIdx.y == 0) {
			bl_norm_L2[blockIdx.x + (gridDim.x * blockIdx.y)] = res_cache[0];
		}
	}
}

__global__ void black_kernel (uint rowmax, uint colmax, const Real * aP,
														  const Real * aW, const Real * aE, const Real * aS,
														  const Real * aN, const Real * b, Real * temp, Real * bl_norm_L2)
{	
	uint row = 1 + (blockIdx.y * blockDim.y) + threadIdx.y;
	uint col = 1 + (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// store residual for block
	__shared__ Real res_cache[block_size];
	
	res_cache[threadIdx.y] = 0.0;
	
	// only black cell if odd
	if ((row + col) % 2 == 1) {
		uint ind = (col - 1) * (rowmax - 2) + (row - 1);
		uint ind2 = col * rowmax + row;
		
		Real res = b[ind] + (aW[ind] * temp[(col - 1) * rowmax + row]
										   + aE[ind] * temp[(col + 1) * rowmax + row]
										   + aS[ind] * temp[col * rowmax + (row - 1)]
										   + aN[ind] * temp[col * rowmax + (row + 1)]);
		
		//temp[ind2] = temp[ind2] * (1.0 - omega) + omega * (res / aP[ind]);
		Real temp_old = temp[ind2];
		Real temp_new = temp_old * (1.0 - omega) + omega * (res / aP[ind]);
		
		temp[ind2] = temp_new;
		res = temp_old - temp_new;
		
		// store squared residual from each thread
		res_cache[threadIdx.y] = res * res;
		
		// synchronize threads in block
		__syncthreads();
		
		// add up squared residuals for block
		uint i = block_size / 2;
		while (i != 0) {
			if (threadIdx.y < i) {
				res_cache[threadIdx.y] += res_cache[threadIdx.y + i];
			}
			__syncthreads();
			i /= 2;
		}
		
		// store block's summed residuals
		if (threadIdx.y == 0) {
			bl_norm_L2[blockIdx.x + (gridDim.x * blockIdx.y)] = res_cache[0];
		}
	}
}

int main (void) {
	
	// size of plate
	Real L = 1.0;
	Real H = 1.0;
	Real width = 0.01;
	
	// thermal conductivity
	Real th_cond = 1.0;
	
	// temperature at top boundary
	Real TN = 1.0;
	
	// SOR iteration tolerance
	Real tol = 1.e-6;
	
	// number of cells in x and y directions
	// including unused boundary cells
	uint num_rows = 4096 + 2;
	uint num_cols = 4096 + 2;
	uint size_temp = num_rows * num_cols;
	uint size = (num_rows - 2) * (num_cols - 2);
	
	// size of cells
	Real dx = L / (num_rows - 2);
	Real dy = H / (num_cols - 2);
	
	// iterations for Red-Black Gauss-Seidel with SOR
	uint iter;
	uint it_max = 1e6;
	
	// allocate memory
	Real *aP, *aW, *aE, *aS, *aN, *b, *temp, *temp_old;
	
	// arrays of coefficients
	aP = (Real *) calloc (size, sizeof(Real));
	aW = (Real *) calloc (size, sizeof(Real));
	aE = (Real *) calloc (size, sizeof(Real));
	aS = (Real *) calloc (size, sizeof(Real));
	aN = (Real *) calloc (size, sizeof(Real));
	
	// RHS
	b = (Real *) calloc (size, sizeof(Real));
	
	// temperature
	temp = (Real *) calloc (size_temp, sizeof(Real));
	temp_old = (Real *) calloc (size_temp, sizeof(Real));
	
	// set coefficients
	fill_coeffs (num_rows - 2, num_cols - 2, th_cond, dx, dy, width, TN, aP, aW, aE, aS, aN, b);
	
	for (uint i = 0; i < size_temp; ++i) {
		temp[i] = 0.0;
		temp_old[i] = 0.0;
	}
	
	// set device
	cudaSetDevice (1);
	
	//////////////////////////////
	// start timer
	//double time, start_time = 0.0;
	//time = walltime(&start_time);
	clock_t start_time = clock();
	//////////////////////////////
	
	// allocate device memory
	Real *aP_d, *aW_d, *aE_d, *aS_d, *aN_d, *b_d, *temp_d;
	
	CUDA_SAFE_CALL (cudaMalloc ((void**) &aP_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &aW_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &aE_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &aS_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &aN_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &b_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &temp_d, size_temp * sizeof(Real)));
	
	// copy to device memory
	CUDA_SAFE_CALL (cudaMemcpy (aP_d, aP, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (aW_d, aW, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (aE_d, aE, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (aS_d, aS, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (aN_d, aN, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (b_d, b, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (temp_d, temp, size_temp * sizeof(Real), cudaMemcpyHostToDevice));
	
	// block and grid dimensions
	
	///////////////////////////////////////
	// naive (no coalescing)
	//dim3 dimBlock (block_size, 1);
	//dim3 dimGrid ((num_rows - 2) / block_size, (num_cols - 2));
	///////////////////////////////////////
	
	///////////////////////////////////////
	// coalescing
	dim3 dimBlock (1, block_size);
	dim3 dimGrid (num_rows - 2, (num_cols - 2) / block_size);
	///////////////////////////////////////
	
	Real *bl_norm_L2, *bl_norm_L2_d;
	
	bl_norm_L2 = (Real *) calloc (dimGrid.x * dimGrid.y, sizeof(Real));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &bl_norm_L2_d, dimGrid.x * dimGrid.y * sizeof(Real)));
		
	// iteration loop
	for (iter = 1; iter <= it_max; ++iter) {
		
		Real norm_L2 = 0.0;
		
		// update red cells
		red_kernel <<<dimGrid, dimBlock>>> (num_rows, num_cols, aP_d, aW_d, aE_d, aS_d, aN_d, b_d, temp_d, bl_norm_L2_d);
		
		CUDA_SAFE_CALL (cudaMemcpy (bl_norm_L2, bl_norm_L2_d, dimGrid.x * dimGrid.y * sizeof(Real), cudaMemcpyDeviceToHost));
		for (uint i = 0; i < (dimGrid.x * dimGrid.y); ++i) {
			norm_L2 += bl_norm_L2[i];
		}
		
		// update black cells
		black_kernel <<<dimGrid, dimBlock>>> (num_rows, num_cols, aP_d, aW_d, aE_d, aS_d, aN_d, b_d, temp_d, bl_norm_L2_d);
		
		// sync threads (needed?)
		//CUDA_SAFE_CALL (cudaThreadSynchronize());
		
		// transfer memory back to host to check for convergence
		//CUDA_SAFE_CALL (cudaMemcpy (temp, temp_d, size_temp * sizeof(Real), cudaMemcpyDeviceToHost));
		
		// transfer residuals back
		CUDA_SAFE_CALL (cudaMemcpy (bl_norm_L2, bl_norm_L2_d, dimGrid.x * dimGrid.y * sizeof(Real), cudaMemcpyDeviceToHost));
		
		// check residual to see if done with iterations
		/*
		for (uint i = 0; i < size_temp; ++i) {
			norm_L2 += (temp[i] - temp_old[i]) * (temp[i] - temp_old[i]);
			temp_old[i] = temp[i];
		}
		*/
		for (uint i = 0; i < (dimGrid.x * dimGrid.y); ++i) {
			norm_L2 += bl_norm_L2[i];
		}
		norm_L2 = sqrt(norm_L2 / size);
		
		// if tolerance has been reached, end SOR iterations
		if (norm_L2 < tol) {
			break;
		}	
	}
	
	// transfer final temperature values back
	CUDA_SAFE_CALL (cudaMemcpy (temp, temp_d, size_temp * sizeof(Real), cudaMemcpyDeviceToHost));
	
	// free device memory
	CUDA_SAFE_CALL (cudaFree(aP_d));
	CUDA_SAFE_CALL (cudaFree(aW_d));
	CUDA_SAFE_CALL (cudaFree(aE_d));
	CUDA_SAFE_CALL (cudaFree(aS_d));
	CUDA_SAFE_CALL (cudaFree(aN_d));
	CUDA_SAFE_CALL (cudaFree(b_d));
	CUDA_SAFE_CALL (cudaFree(temp_d));
	
	CUDA_SAFE_CALL (cudaFree(bl_norm_L2_d));
	
	/////////////////////////////////
	// end timer
	//time = walltime(&time);
	clock_t end_time = clock();
	/////////////////////////////////
	
	printf("GPU\nIterations: %i\n", iter);
	printf("Time: %f\n", (end_time - start_time) / (double)CLOCKS_PER_SEC);
	
	FILE * pfile;
	pfile = fopen("temp_gpu.dat", "w");
	
	if (pfile != NULL) {
		fprintf(pfile, "#x\ty\ttemp(K)\n");
		
		for (uint row = 1; row < num_rows - 1; ++row) {
			for (uint col = 1; col < num_cols - 1; ++col) {
				uint ind = col * num_rows + row;
				Real x_pos = (col - 1) * dx + (dx / 2);
				Real y_pos = (row - 1) * dy + (dy / 2);
				fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp[ind]);
			}
			fprintf(pfile, "\n");
		}
	}
	fclose(pfile);
	
	free(aP);
	free(aW);
	free(aE);
	free(aS);
	free(aN);
	free(b);
	free(temp);
	free(temp_old);
	
	free(bl_norm_L2);
	
	cudaDeviceReset();
	
	return 0;
}
