/** GPU Laplace solver using optimized red-black Gauss–Seidel with SOR solver
 * \file main_cpu_opt.c
 *
 * \author Kyle E. Niemeyer
 * \date 09/21/2012
 *
 * Solves Laplace's equation in 2D (e.g., heat conduction in a rectangular plate)
 * on GPU using CUDA with the red-black Gauss–Seidel with sucessive overrelaxation
 * (SOR) that has been "optimized". This means that the red and black kernels 
 * only loop overtheir respective cells, instead of over all cells and skipping
 * even/odd cells. This requires separate arrays for red and black cells.
 * 
 * Boundary conditions:
 * T = 0 at x = 0, x = L, y = 0
 * T = TN at y = H
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// CUDA libraries
#include <cuda.h>
#include <cutil.h>

/** Problem size along one side; total number of cells is this squared */
#define NUM 2048

// block size
#define BLOCK_SIZE 128

/** Double precision */
//#define DOUBLE

#ifdef DOUBLE
	#define Real double
#else
	#define Real float
#endif

/** Use texture memory */
//#define TEXTURE

/** Use atomic operations to calculate residual, only for SINGLE PRECISION */
//#define ATOMIC

#if defined (ATOMIC) && defined (DOUBLE)
# error double precision atomic operations not supported
#endif

typedef unsigned int uint;

/** SOR relaxation parameter */
const Real omega = 1.85;

#ifdef TEXTURE
#ifdef DOUBLE
texture<int2,1> aP_t;
texture<int2,1> aW_t;
texture<int2,1> aE_t;
texture<int2,1> aS_t;
texture<int2,1> aN_t;
texture<int2,1> b_t;

static __inline__ __device__ double get_tex(texture<int2, 1> tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
	return __hiloint2double(v.y, v.x);
}
#else
texture<float> aP_t;
texture<float> aW_t;
texture<float> aE_t;
texture<float> aS_t;
texture<float> aN_t;
texture<float> b_t;

static __inline__ __device__ float get_tex(texture<float> tex, int i)
{
	return tex1Dfetch(tex, i);
}
#endif
#endif

///////////////////////////////////////////////////////////////////////////////

/** Function to evaluate coefficient matrix and right-hand side vector.
 * 
 * \param[in]		rowmax		number of rows
 * \param[in]		colmax		number of columns
 * \param[in]		th_cond		thermal conductivity
 * \param[in]		dx				grid size in x dimension (uniform)
 * \param[in]		dy				grid size in y dimension (uniform)
 * \param[in]		width			width of plate (z dimension)
 * \param[in]		TN				temperature at top boundary
 * \param[out]	aP				array of self coefficients
 * \param[out]	aW				array of west neighbor coefficients
 * \param[out]	aE				array of east neighbor coefficients
 * \param[out]	aS				array of south neighbor coefficients
 * \param[out]	aN				array of north neighbor coefficients
 * \param[out]	b					right-hand side array
 */
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
} // end fill_coeffs

///////////////////////////////////////////////////////////////////////////////

/** Function to update temperature for red cells
 * 
 * \param[in]			aP					array of self coefficients
 * \param[in]			aW					array of west neighbor coefficients
 * \param[in]			aE					array of east neighbor coefficients
 * \param[in]			aS					array of south neighbor coefficients
 * \param[in]			aN					array of north neighbor coefficients
 * \param[in]			b						right-hand side array
 * \param[in]			temp_black	temperatures of black cells, constant in this function
 * \param[inout]	temp_red		temperatures of red cells
 * \param[out]		bl_norm_L2	array with residual information for blocks
 */
#ifdef TEXTURE
__global__ void red_kernel (const Real * temp_black, Real * temp_red, Real * bl_norm_L2)
#else
__global__ void red_kernel (const Real * aP, const Real * aW, const Real * aE,
														const Real * aS, const Real * aN, const Real * b,
														const Real * temp_black, Real * temp_red,
														Real * bl_norm_L2)
#endif
{	
	uint row = 1 + (blockIdx.y * blockDim.y) + threadIdx.y;
	uint col = 1 + (blockIdx.x * blockDim.x) + threadIdx.x;

	// store residual for block
	__shared__ Real res_cache[BLOCK_SIZE];
	res_cache[threadIdx.y] = 0.0;
	
	uint ind_red = col * ((NUM >> 1) + 2) + row;  		// local (red) index
	uint ind = 2 * row - (col & 1) - 1 + NUM * (col - 1);	// global index
	
	Real temp_old = temp_red[ind_red];
	
	#ifdef TEXTURE
	Real res = get_tex(b_t, ind) 
					 + (get_tex(aW_t, ind) * temp_black[row + (col - 1) * ((NUM >> 1) + 2)]
				    + get_tex(aE_t, ind) * temp_black[row + (col + 1) * ((NUM >> 1) + 2)]
				    + get_tex(aS_t, ind) * temp_black[row - (col & 1) + col * ((NUM >> 1) + 2)]
				    + get_tex(aN_t, ind) * temp_black[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)]);
	
	Real temp_new = temp_old * (1.0 - omega) + omega * (res / get_tex(aP_t, ind));
	#else
	Real res = b[ind]
					 + (aW[ind] * temp_black[row + (col - 1) * ((NUM >> 1) + 2)]
				    + aE[ind] * temp_black[row + (col + 1) * ((NUM >> 1) + 2)]
				    + aS[ind] * temp_black[row - (col & 1) + col * ((NUM >> 1) + 2)]
				    + aN[ind] * temp_black[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)]);
	
	Real temp_new = temp_old * (1.0 - omega) + omega * (res / aP[ind]);
	#endif
	
	temp_red[ind_red] = temp_new;
	res = temp_old - temp_new;
	
	// store squared residual from each thread
	res_cache[threadIdx.y] = res * res;
	
	// synchronize threads in block
	__syncthreads();
	
	// add up squared residuals for block
	uint i = BLOCK_SIZE >> 1;
	while (i != 0) {
		if (threadIdx.y < i) {
			res_cache[threadIdx.y] += res_cache[threadIdx.y + i];
		}
		__syncthreads();
		i = i >> 1;
	}
	
	// store block's summed residuals
	if (threadIdx.y == 0) {
		#ifdef ATOMIC
		atomicAdd (bl_norm_L2, res_cache[0]);
		#else
		bl_norm_L2[blockIdx.x + (gridDim.x * blockIdx.y)] = res_cache[0];
		#endif
	}
} // end red_kernel

///////////////////////////////////////////////////////////////////////////////

/** Function to update temperature for black cells
 * 
 * \param[in]			aP					array of self coefficients
 * \param[in]			aW					array of west neighbor coefficients
 * \param[in]			aE					array of east neighbor coefficients
 * \param[in]			aS					array of south neighbor coefficients
 * \param[in]			aN					array of north neighbor coefficients
 * \param[in]			b						right-hand side array
 * \param[in]			temp_red		temperatures of red cells, constant in this function
 * \param[inout]	temp_black	temperatures of black cells
 * \param[out]		bl_norm_L2	array with residual information for blocks
 */
#ifdef TEXTURE
__global__ void black_kernel (const Real * temp_red, Real * temp_black, Real * bl_norm_L2)
#else
__global__ void black_kernel (const Real * aP, const Real * aW, const Real * aE,
														  const Real * aS, const Real * aN, const Real * b,
															const Real * temp_red, Real * temp_black, 
															Real * bl_norm_L2)
#endif
{	
	uint row = 1 + (blockIdx.y * blockDim.y) + threadIdx.y;
	uint col = 1 + (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// store residual for block
	__shared__ Real res_cache[BLOCK_SIZE];
	res_cache[threadIdx.y] = 0.0;
	
	uint ind_black = col * ((NUM >> 1) + 2) + row;  					// local (black) index
	uint ind = 2 * row - ((col + 1) & 1) - 1 + NUM * (col - 1);	// global index
	
	Real temp_old = temp_black[ind_black];
	#ifdef TEXTURE
	Real res = get_tex(b_t, ind)
	 				 + (get_tex(aW_t, ind) * temp_red[row + (col - 1) * ((NUM >> 1) + 2)]
				    + get_tex(aE_t, ind) * temp_red[row + (col + 1) * ((NUM >> 1) + 2)]
				    + get_tex(aS_t, ind) * temp_red[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)]
				    + get_tex(aN_t, ind) * temp_red[row + (col & 1) + col * ((NUM >> 1) + 2)]);
	
	Real temp_new = temp_old * (1.0 - omega) + omega * (res / get_tex(aP_t, ind));
	#else
	Real res = b[ind]
	 				 + (aW[ind] * temp_red[row + (col - 1) * ((NUM >> 1) + 2)]
				    + aE[ind] * temp_red[row + (col + 1) * ((NUM >> 1) + 2)]
				    + aS[ind] * temp_red[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)]
				    + aN[ind] * temp_red[row + (col & 1) + col * ((NUM >> 1) + 2)]);
	
	Real temp_new = temp_old * (1.0 - omega) + omega * (res / aP[ind]);
	#endif
	
	temp_black[ind_black] = temp_new;
	res = temp_old - temp_new;
	
	// store squared residual from each thread
	res_cache[threadIdx.y] = res * res;
	
	// synchronize threads in block
	__syncthreads();
	
	// add up squared residuals for block
	uint i = BLOCK_SIZE >> 1;
	while (i != 0) {
		if (threadIdx.y < i) {
			res_cache[threadIdx.y] += res_cache[threadIdx.y + i];
		}
		__syncthreads();
		i = i >> 1;
	}
	
	// store block's summed residuals
	if (threadIdx.y == 0) {
		#ifdef ATOMIC
		atomicAdd (bl_norm_L2, res_cache[0]);
		#else
		bl_norm_L2[blockIdx.x + (gridDim.x * blockIdx.y)] = res_cache[0];
		#endif
	}
} // end black_kernel

///////////////////////////////////////////////////////////////////////////////

/** Main function that solves Laplace's equation in 2D (heat conduction in plate)
 * 
 * Contains iteration loop for red-black Gauss-Seidel with SOR GPU kernels
 */
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
	uint num_rows = (NUM / 2) + 2;
	uint num_cols = NUM + 2;
	uint size_temp = num_rows * num_cols;
	uint size = NUM * NUM;
	
	// size of cells
	Real dx = L / NUM;
	Real dy = H / NUM;
	
	// iterations for Red-Black Gauss-Seidel with SOR
	uint iter;
	uint it_max = 1e6;
	
	// allocate memory
	Real *aP, *aW, *aE, *aS, *aN, *b;
	Real *temp_red, *temp_black;
	
	// arrays of coefficients
	aP = (Real *) calloc (size, sizeof(Real));
	aW = (Real *) calloc (size, sizeof(Real));
	aE = (Real *) calloc (size, sizeof(Real));
	aS = (Real *) calloc (size, sizeof(Real));
	aN = (Real *) calloc (size, sizeof(Real));
	
	// RHS
	b = (Real *) calloc (size, sizeof(Real));
	
	// temperature arrays
	temp_red = (Real *) calloc (size_temp, sizeof(Real));
	temp_black = (Real *) calloc (size_temp, sizeof(Real));
	
	// set coefficients
	fill_coeffs (NUM, NUM, th_cond, dx, dy, width, TN, aP, aW, aE, aS, aN, b);
	
	for (uint i = 0; i < size_temp; ++i) {
		temp_red[i] = 0.0;
		temp_black[i] = 0.0;
	}
	
	// set device
	cudaSetDevice (1);
	
	//////////////////////////////
	// start timer
	////double time, start_time = 0.0;
	////time = walltime(&start_time);
	clock_t start_time = clock();
	//////////////////////////////
	
	// allocate device memory
	Real *aP_d, *aW_d, *aE_d, *aS_d, *aN_d, *b_d;
	Real *temp_red_d, *temp_black_d;
	
	CUDA_SAFE_CALL (cudaMalloc ((void**) &aP_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &aW_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &aE_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &aS_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &aN_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &b_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &temp_red_d, size_temp * sizeof(Real)));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &temp_black_d, size_temp * sizeof(Real)));
	
	// copy to device memory
	CUDA_SAFE_CALL (cudaMemcpy (aP_d, aP, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (aW_d, aW, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (aE_d, aE, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (aS_d, aS, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (aN_d, aN, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (b_d, b, size * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (temp_red_d, temp_red, size_temp * sizeof(Real), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL (cudaMemcpy (temp_black_d, temp_black, size_temp * sizeof(Real), cudaMemcpyHostToDevice));
	
	#ifdef TEXTURE
	// bind to textures
	CUDA_SAFE_CALL (cudaBindTexture (NULL, aP_t, aP_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaBindTexture (NULL, aW_t, aW_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaBindTexture (NULL, aE_t, aE_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaBindTexture (NULL, aS_t, aS_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaBindTexture (NULL, aN_t, aN_d, size * sizeof(Real)));
	CUDA_SAFE_CALL (cudaBindTexture (NULL, b_t, b_d, size * sizeof(Real)));
	#endif
	
	// block and grid dimensions
	
	///////////////////////////////////////
	// naive (no coalescing)
	//dim3 dimBlock (BLOCK_SIZE, 1);
	//dim3 dimGrid ((num_rows - 2) / BLOCK_SIZE, (num_cols - 2));
	///////////////////////////////////////
	
	///////////////////////////////////////
	// coalescing
	dim3 dimBlock (1, BLOCK_SIZE);
	dim3 dimGrid (NUM, NUM / (2 * BLOCK_SIZE));
	///////////////////////////////////////
	
	// residual variables
	Real *bl_norm_L2, *bl_norm_L2_d;
		
	#ifdef ATOMIC
	// single value, using atomic operations to sum
	bl_norm_L2 = (Real *) malloc (sizeof(Real));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &bl_norm_L2_d, sizeof(Real)));
	#else
	bl_norm_L2 = (Real *) calloc (dimGrid.x * dimGrid.y, sizeof(Real));
	CUDA_SAFE_CALL (cudaMalloc ((void**) &bl_norm_L2_d, dimGrid.x * dimGrid.y * sizeof(Real)));
	#endif
		
	// iteration loop
	for (iter = 1; iter <= it_max; ++iter) {
		
		Real norm_L2 = 0.0;
		
		#ifdef ATOMIC
		// set device value to zero
		*bl_norm_L2 = 0.0;
		CUDA_SAFE_CALL (cudaMemcpy (bl_norm_L2_d, bl_norm_L2, sizeof(Real), cudaMemcpyHostToDevice));
		#endif
		
		// update red cells
		#ifdef TEXTURE
		red_kernel <<<dimGrid, dimBlock>>> (temp_black_d, temp_red_d, bl_norm_L2_d);
		#else
		red_kernel <<<dimGrid, dimBlock>>> (aP_d, aW_d, aE_d, aS_d, aN_d, b_d, temp_black_d, temp_red_d, bl_norm_L2_d);
		#endif
		
		// transfer residual value(s) back to CPU
		#ifdef ATOMIC
		CUDA_SAFE_CALL (cudaMemcpy (bl_norm_L2, bl_norm_L2_d, sizeof(Real), cudaMemcpyDeviceToHost));
		#else
		CUDA_SAFE_CALL (cudaMemcpy (bl_norm_L2, bl_norm_L2_d, dimGrid.x * dimGrid.y * sizeof(Real), cudaMemcpyDeviceToHost));
		#endif
		
		// add red cell contributions to residual
		#ifndef ATOMIC
		for (uint i = 0; i < (dimGrid.x * dimGrid.y); ++i) {
			norm_L2 += bl_norm_L2[i];
		}
		#endif
		norm_L2 += *bl_norm_L2;
		
		#ifdef ATOMIC
		// set device value to zero
		*bl_norm_L2 = 0.0;
		CUDA_SAFE_CALL (cudaMemcpy (bl_norm_L2_d, bl_norm_L2, sizeof(Real), cudaMemcpyHostToDevice));
		#endif
		
		// update black cells
		#ifdef TEXTURE
		black_kernel <<<dimGrid, dimBlock>>> (temp_red_d, temp_black_d, bl_norm_L2_d);
		#else
		black_kernel <<<dimGrid, dimBlock>>> (aP_d, aW_d, aE_d, aS_d, aN_d, b_d, temp_red_d, temp_black_d, bl_norm_L2_d);
		#endif
		
		// sync threads (needed?)
		//CUDA_SAFE_CALL (cudaThreadSynchronize());
		
		// transfer residual value(s) back to CPU
		#ifdef ATOMIC
		CUDA_SAFE_CALL (cudaMemcpy (bl_norm_L2, bl_norm_L2_d, sizeof(Real), cudaMemcpyDeviceToHost));
		#else
		CUDA_SAFE_CALL (cudaMemcpy (bl_norm_L2, bl_norm_L2_d, dimGrid.x * dimGrid.y * sizeof(Real), cudaMemcpyDeviceToHost));
		#endif
		
		// add black cell contributions to residual
		#ifndef ATOMIC
		for (uint i = 0; i < (dimGrid.x * dimGrid.y); ++i) {
			norm_L2 += bl_norm_L2[i];
		}
		#endif
		norm_L2 += *bl_norm_L2;
		
		// calculate residual
		norm_L2 = sqrt(norm_L2 / size);
		
		// if tolerance has been reached, end SOR iterations
		if (norm_L2 < tol) {
			break;
		}	
	}
	
	// transfer final temperature values back
	CUDA_SAFE_CALL (cudaMemcpy (temp_red, temp_red_d, size_temp * sizeof(Real), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL (cudaMemcpy (temp_black, temp_red_d, size_temp * sizeof(Real), cudaMemcpyDeviceToHost));
	
	// free device memory
	CUDA_SAFE_CALL (cudaFree(aP_d));
	CUDA_SAFE_CALL (cudaFree(aW_d));
	CUDA_SAFE_CALL (cudaFree(aE_d));
	CUDA_SAFE_CALL (cudaFree(aS_d));
	CUDA_SAFE_CALL (cudaFree(aN_d));
	CUDA_SAFE_CALL (cudaFree(b_d));
	CUDA_SAFE_CALL (cudaFree(temp_red_d));
	CUDA_SAFE_CALL (cudaFree(temp_black_d));
	
	CUDA_SAFE_CALL (cudaFree(bl_norm_L2_d));
	
	#ifdef TEXTURE
	// unbind textures
	CUDA_SAFE_CALL (cudaUnbindTexture(aP_t));
	CUDA_SAFE_CALL (cudaUnbindTexture(aW_t));
	CUDA_SAFE_CALL (cudaUnbindTexture(aE_t));
	CUDA_SAFE_CALL (cudaUnbindTexture(aS_t));
	CUDA_SAFE_CALL (cudaUnbindTexture(aN_t));
	CUDA_SAFE_CALL (cudaUnbindTexture(b_t));
	#endif
	
	/////////////////////////////////
	// end timer
	//time = walltime(&time);
	clock_t end_time = clock();
	/////////////////////////////////
	
	printf("GPU\nIterations: %i\n", iter);
	printf("Time: %f\n", (end_time - start_time) / (double)CLOCKS_PER_SEC);
	
	// print temperature data to file
	FILE * pfile;
	pfile = fopen("temp_gpu.dat", "w");
	
	if (pfile != NULL) {
		fprintf(pfile, "#x\ty\ttemp(K)\n");
		
		for (uint row = 1; row < NUM + 1; ++row) {
			for (uint col = 1; col < NUM + 1; ++col) {
				Real x_pos = (col - 1) * dx + (dx / 2);
				Real y_pos = (row - 1) * dy + (dy / 2);
				
				if ((row + col) % 2 == 0) {
					// even, so red cell
					uint ind = col * num_rows + (row + (col % 2)) / 2;
					fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_red[ind]);
				} else {
					// odd, so black cell
					uint ind = col * num_rows + (row + ((col + 1) % 2)) / 2;
					fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_black[ind]);
				}	
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
	free(temp_red);
	free(temp_black);
	#ifndef ATOMIC
	free(bl_norm_L2);
	#endif
	
	cudaDeviceReset();
	
	return 0;
}
