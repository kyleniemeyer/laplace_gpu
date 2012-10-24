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

#include "timer.h"

// CUDA libraries
#include <cuda.h>
#include <helper_cuda.h>

/** Problem size along one side; total number of cells is this squared */
#define NUM 2048

// block size
#define BLOCK_SIZE 128

/** Double precision */
#define DOUBLE

#ifdef DOUBLE
	#define Real double
	#define ZERO 0.0
	#define ONE 1.0
	#define TWO 2.0

	/** SOR relaxation parameter */
	const Real omega = 1.85;
#else
	#define Real float
	#define ZERO 0.0f
	#define ONE 1.0f
	#define TWO 2.0f

	/** SOR relaxation parameter */
	const Real omega = 1.85f;
#endif

/** Arrange global memory for coalescing */
#define COALESCE

/** Split temperature into red and black arrays */
#define MEMOPT

/** Use shared memory to get residual */
//#define SHARED

/** Use texture memory */
//#define TEXTURE

/** Use atomic operations to calculate residual, only for SINGLE PRECISION */
//#define ATOMIC

#if defined (ATOMIC) && defined (DOUBLE)
# error double precision atomic operations not supported
#endif

#ifdef TEXTURE
	#ifdef DOUBLE
		texture<int2,1> aP_t;
		texture<int2,1> aW_t;
		texture<int2,1> aE_t;
		texture<int2,1> aS_t;
		texture<int2,1> aN_t;
		texture<int2,1> b_t;

		static __inline__ __device__ double get_tex (texture<int2, 1> tex, int i)
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

		static __inline__ __device__ float get_tex (texture<float> tex, int i)
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
void fill_coeffs (int rowmax, int colmax, Real th_cond, Real dx, Real dy,
				  				Real width, Real TN, Real * aP, Real * aW, Real * aE, 
				  				Real * aS, Real * aN, Real * b)
{
	int col, row;
	for (col = 0; col < colmax; ++col) {
		for (row = 0; row < rowmax; ++row) {
			int ind = col * rowmax + row;
			
			b[ind] = ZERO;
			Real SP = ZERO;
			
			if (col == 0) {
				// left BC: temp = 0
				aW[ind] = ZERO;
				SP = -TWO * th_cond * width * dy / dx;
			} else {
				aW[ind] = th_cond * width * dy / dx;
			}
			
			if (col == colmax - 1) {
				// right BC: temp = 0
				aE[ind] = ZERO;
				SP = -TWO * th_cond * width * dy / dx;
			} else {
				aE[ind] = th_cond * width * dy / dx;
			}
			
			if (row == 0) {
				// bottom BC: temp = 0
				aS[ind] = ZERO;
				SP = -TWO * th_cond * width * dx / dy;
			} else {
				aS[ind] = th_cond * width * dx / dy;
			}
			
			if (row == rowmax - 1) {
				// top BC: temp = TN
				aN[ind] = ZERO;
				b[ind] = TWO * th_cond * width * dx * TN / dy;
				SP = -TWO * th_cond * width * dx / dy;
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
__global__ void red_kernel (const Real * temp_black, Real * temp_red, Real * norm_L2)
#else
__global__ void red_kernel (const Real * aP, const Real * aW, const Real * aE,
														const Real * aS, const Real * aN, const Real * b,
														const Real * temp_black, Real * temp_red,
														Real * norm_L2)
#endif
{	
	int row = 1 + (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = 1 + (blockIdx.x * blockDim.x) + threadIdx.x;

	// store residual for block
	#ifdef SHARED
		__shared__ Real res_cache[BLOCK_SIZE];
		res_cache[threadIdx.y] = ZERO;
	#endif
	
	#ifdef MEMOPT
		int ind_red = col * ((NUM >> 1) + 2) + row;  		// local (red) index
		int ind = 2 * row - (col & 1) - 1 + NUM * (col - 1);	// global index
	#else
	if ((row + col) % 2 == 0) {
		int ind_red = (col * (NUM + 2)) + row;
		int ind = ((col - 1) * NUM ) + row - 1;
	#endif
	
	Real temp_old = temp_red[ind_red];
	
	#if defined(TEXTURE) && defined(MEMOPT)
		Real res = get_tex(b_t, ind)
		 				+ (get_tex(aW_t, ind) * temp_black[row + (col - 1) * ((NUM >> 1) + 2)]
					   + get_tex(aE_t, ind) * temp_black[row + (col + 1) * ((NUM >> 1) + 2)]
					   + get_tex(aS_t, ind) * temp_black[row - (col & 1) + col * ((NUM >> 1) + 2)]
					   + get_tex(aN_t, ind) * temp_black[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)]);
		
		Real temp_new = temp_old * (ONE - omega) + omega * (res / get_tex(aP_t, ind));
	#elif defined(TEXTURE) && !defined(MEMOPT)
		Real res = get_tex(b_t, ind)
		 				+ (get_tex(aW_t, ind) * temp_black[row + (col - 1) * (NUM + 2)]
					   + get_tex(aE_t, ind) * temp_black[row + (col + 1) * (NUM + 2)]
					   + get_tex(aS_t, ind) * temp_black[row - 1 + col * (NUM + 2)]
					   + get_tex(aN_t, ind) * temp_black[row + 1 + col * (NUM + 2)]);
		
		Real temp_new = temp_old * (ONE - omega) + omega * (res / get_tex(aP_t, ind));
	#elif !defined(TEXTURE) && defined(MEMOPT)
		Real res = b[ind]
					 + (aW[ind] * temp_black[row + (col - 1) * ((NUM >> 1) + 2)]
				    + aE[ind] * temp_black[row + (col + 1) * ((NUM >> 1) + 2)]
				    + aS[ind] * temp_black[row - (col & 1) + col * ((NUM >> 1) + 2)]
				    + aN[ind] * temp_black[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)]);
		
		Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);
	#else
		// neither TEXTURE nor MEMOPT defined
		Real res = b[ind]
 				 	 + (aW[ind] * temp_black[row + (col - 1) * (NUM + 2)]
			    	+ aE[ind] * temp_black[row + (col + 1) * (NUM + 2)]
			    	+ aS[ind] * temp_black[row - 1 + col * (NUM + 2)]
			    	+ aN[ind] * temp_black[row + 1 + col * (NUM + 2)]);
		
		Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);
	#endif
	
	temp_red[ind_red] = temp_new;
	res = temp_new - temp_old;
	
	#ifdef SHARED
		// store squared residual from each thread in block
		res_cache[threadIdx.y] = res * res;
		
		// synchronize threads in block
		__syncthreads();
		
		// add up squared residuals for block
		int i = BLOCK_SIZE >> 1;
		while (i != 0) {
			if (threadIdx.y < i) {
				res_cache[threadIdx.y] += res_cache[threadIdx.y + i];
			}
			__syncthreads();
			i >>= 1;
		}
		
		// store block's summed residuals
		if (threadIdx.y == 0) {
			#ifdef ATOMIC
				atomicAdd (norm_L2, res_cache[0]);
			#else
				norm_L2[blockIdx.x + (gridDim.x * blockIdx.y)] = res_cache[0];
			#endif
		}
	#else
		norm_L2[ind_red] = res * res;
	#endif

	#ifndef MEMOPT
	}
	#endif
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
__global__ void black_kernel (const Real * temp_red, Real * temp_black, Real * norm_L2)
#else
__global__ void black_kernel (const Real * aP, const Real * aW, const Real * aE,
								const Real * aS, const Real * aN, const Real * b,
								const Real * temp_red, Real * temp_black, 
								Real * norm_L2)
#endif
{	
	int row = 1 + (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = 1 + (blockIdx.x * blockDim.x) + threadIdx.x;
	
	#ifdef SHARED
		// store residual for block
		__shared__ Real res_cache[BLOCK_SIZE];
		res_cache[threadIdx.y] = ZERO;
	#endif
	
	#ifdef MEMOPT	
		int ind_black = col * ((NUM >> 1) + 2) + row;  					// local (black) index
		int ind = 2 * row - ((col + 1) & 1) - 1 + NUM * (col - 1);	// global index
	#else
	if ((row + col) % 2 == 1) {
		int ind_black = (col * (NUM + 2)) + row;
		int ind = ((col - 1) * NUM ) + row - 1;
	#endif
	
	Real temp_old = temp_black[ind_black];

	#if defined(TEXTURE) && defined(MEMOPT)
		Real res = get_tex(b_t, ind)
		 				+ (get_tex(aW_t, ind) * temp_red[row + (col - 1) * ((NUM >> 1) + 2)]
					   + get_tex(aE_t, ind) * temp_red[row + (col + 1) * ((NUM >> 1) + 2)]
					   + get_tex(aS_t, ind) * temp_red[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)]
					   + get_tex(aN_t, ind) * temp_red[row + (col & 1) + col * ((NUM >> 1) + 2)]);
		
		Real temp_new = temp_old * (ONE - omega) + omega * (res / get_tex(aP_t, ind));
	#elif defined(TEXTURE) && !defined(MEMOPT)
		Real res = get_tex(b_t, ind)
		 				+ (get_tex(aW_t, ind) * temp_red[row + (col - 1) * (NUM + 2)]
					   + get_tex(aE_t, ind) * temp_red[row + (col + 1) * (NUM + 2)]
					   + get_tex(aS_t, ind) * temp_red[row - 1 + col * (NUM + 2)]
					   + get_tex(aN_t, ind) * temp_red[row + 1 + col * (NUM + 2)]);
		
		Real temp_new = temp_old * (ONE - omega) + omega * (res / get_tex(aP_t, ind));
	#elif !defined(TEXTURE) && defined(MEMOPT)
		Real res = b[ind]
		 			 + (aW[ind] * temp_red[row + (col - 1) * ((NUM >> 1) + 2)]
					  + aE[ind] * temp_red[row + (col + 1) * ((NUM >> 1) + 2)]
					  + aS[ind] * temp_red[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)]
					  + aN[ind] * temp_red[row + (col & 1) + col * ((NUM >> 1) + 2)]);
		
		Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);
	#else
		// neither TEXTURE nor MEMOPT defined
		Real res = b[ind]
 				 	 + (aW[ind] * temp_red[row + (col - 1) * (NUM + 2)]
			    	+ aE[ind] * temp_red[row + (col + 1) * (NUM + 2)]
			    	+ aS[ind] * temp_red[row - 1 + col * (NUM + 2)]
			    	+ aN[ind] * temp_red[row + 1 + col * (NUM + 2)]);
		
		Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);
	#endif
	
	temp_black[ind_black] = temp_new;
	res = temp_new - temp_old;
	
	#ifdef SHARED
		// store squared residual from each thread in block
		res_cache[threadIdx.y] = res * res;
		
		// synchronize threads in block
		__syncthreads();
		
		// add up squared residuals for block
		int i = BLOCK_SIZE >> 1;
		while (i != 0) {
			if (threadIdx.y < i) {
				res_cache[threadIdx.y] += res_cache[threadIdx.y + i];
			}
			__syncthreads();
			i >>= 1;
		}
		
		// store block's summed residuals
		if (threadIdx.y == 0) {
			#ifdef ATOMIC
				atomicAdd (norm_L2, res_cache[0]);
			#else
				norm_L2[blockIdx.x + (gridDim.x * blockIdx.y)] = res_cache[0];
			#endif
		}
	#else
		norm_L2[ind_black] = res * res;
	#endif

	#ifndef MEMOPT
		}
	#endif
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
	#ifdef MEMOPT
		int num_rows = (NUM / 2) + 2;
	#else
		int num_rows = NUM + 2;
	#endif
	int num_cols = NUM + 2;
	int size_temp = num_rows * num_cols;
	int size = NUM * NUM;
	
	// size of cells
	Real dx = L / NUM;
	Real dy = H / NUM;
	
	// iterations for Red-Black Gauss-Seidel with SOR
	int iter;
	int it_max = 1e6;
	
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
	
	int i;
	for (i = 0; i < size_temp; ++i) {
		temp_red[i] = ZERO;
		temp_black[i] = ZERO;
	}

	//////////////////////////////
	// block and grid dimensions
	//////////////////////////////
	
	#ifdef COALESCE
		///////////////////////////////////////
		// coalescing
		dim3 dimBlock (1, BLOCK_SIZE);
		#ifdef MEMOPT
			dim3 dimGrid (NUM, NUM / (2 * BLOCK_SIZE));
		#else
			dim3 dimGrid (NUM, NUM / BLOCK_SIZE);
		#endif
		///////////////////////////////////////
	#else
		///////////////////////////////////////
		// naive (no coalescing)
		dim3 dimBlock (BLOCK_SIZE, 1);
		#ifdef MEMOPT
			dim3 dimGrid (NUM / BLOCK_SIZE, NUM / 2);
		#else
			dim3 dimGrid (NUM / BLOCK_SIZE, NUM);
		#endif
		///////////////////////////////////////
	#endif

	// residual
	Real *bl_norm_L2;

	#ifdef SHARED
		#ifdef ATOMIC
			int size_norm = 1;
			// single value, using atomic operations to sum
		#else
			// one value for each block
			int size_norm = dimGrid.x * dimGrid.y;
		#endif
	#else
		// one for each temperature value
		int size_norm = size_temp;
	#endif
	bl_norm_L2 = (Real *) calloc (size_norm, sizeof(Real));
	for (i = 0; i < size_norm; ++i) {
		bl_norm_L2[i] = ZERO;
	}
	
	// set device
	checkCudaErrors(cudaSetDevice (1));

	// print problem info
	printf("Problem size: %d x %d \n", NUM, NUM);
	
	//////////////////////////////
	// start timer
	//clock_t start_time = clock();
	StartTimer();
	//////////////////////////////
	
	// allocate device memory
	Real *aP_d, *aW_d, *aE_d, *aS_d, *aN_d, *b_d;
	Real *temp_red_d;
	#ifdef MEMOPT
		Real *temp_black_d;
	#endif
	
	cudaMalloc ((void**) &aP_d, size * sizeof(Real));
	cudaMalloc ((void**) &aW_d, size * sizeof(Real));
	cudaMalloc ((void**) &aE_d, size * sizeof(Real));
	cudaMalloc ((void**) &aS_d, size * sizeof(Real));
	cudaMalloc ((void**) &aN_d, size * sizeof(Real));
	cudaMalloc ((void**) &b_d, size * sizeof(Real));
	cudaMalloc ((void**) &temp_red_d, size_temp * sizeof(Real));
	#ifdef MEMOPT
		cudaMalloc ((void**) &temp_black_d, size_temp * sizeof(Real));
	#endif
	
	// copy to device memory
	cudaMemcpy (aP_d, aP, size * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMemcpy (aW_d, aW, size * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMemcpy (aE_d, aE, size * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMemcpy (aS_d, aS, size * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMemcpy (aN_d, aN, size * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMemcpy (b_d, b, size * sizeof(Real), cudaMemcpyHostToDevice);
	cudaMemcpy (temp_red_d, temp_red, size_temp * sizeof(Real), cudaMemcpyHostToDevice);
	#ifdef MEMOPT
		cudaMemcpy (temp_black_d, temp_black, size_temp * sizeof(Real), cudaMemcpyHostToDevice);
	#endif
	
	#ifdef TEXTURE
		// bind to textures
		cudaBindTexture (NULL, aP_t, aP_d, size * sizeof(Real));
		cudaBindTexture (NULL, aW_t, aW_d, size * sizeof(Real));
		cudaBindTexture (NULL, aE_t, aE_d, size * sizeof(Real));
		cudaBindTexture (NULL, aS_t, aS_d, size * sizeof(Real));
		cudaBindTexture (NULL, aN_t, aN_d, size * sizeof(Real));
		cudaBindTexture (NULL, b_t, b_d, size * sizeof(Real));
	#endif
	
	// residual
	Real *bl_norm_L2_d;	
	cudaMalloc ((void**) &bl_norm_L2_d, size_norm * sizeof(Real));
	#ifndef SHARED
		cudaMemcpy (bl_norm_L2_d, bl_norm_L2, size_norm * sizeof(Real), cudaMemcpyHostToDevice);
	#endif
		
	// iteration loop
	for (iter = 1; iter <= it_max; ++iter) {
		
		Real norm_L2 = ZERO;
		
		#ifdef ATOMIC
			// set device value to zero
			*bl_norm_L2 = ZERO;
			cudaMemcpy (bl_norm_L2_d, bl_norm_L2, sizeof(Real), cudaMemcpyHostToDevice);
		#endif
		
		// update red cells
		#if defined(TEXTURE) && defined(MEMOPT)
			red_kernel <<<dimGrid, dimBlock>>> (temp_black_d, temp_red_d, bl_norm_L2_d);
		#elif defined(TEXTURE) && !defined(MEMOPT)
			red_kernel <<<dimGrid, dimBlock>>> (temp_red_d, temp_red_d, bl_norm_L2_d);
		#elif !defined(TEXTURE) && defined(MEMOPT)
			red_kernel <<<dimGrid, dimBlock>>> (aP_d, aW_d, aE_d, aS_d, aN_d, b_d, temp_black_d, temp_red_d, bl_norm_L2_d);
		#else // neither defined
			red_kernel <<<dimGrid, dimBlock>>> (aP_d, aW_d, aE_d, aS_d, aN_d, b_d, temp_red_d, temp_red_d, bl_norm_L2_d);
		#endif
		
		// transfer residual value(s) back to CPU
		#if !defined(ATOMIC) && defined(MEMOPT)
			cudaMemcpy (bl_norm_L2, bl_norm_L2_d, size_norm * sizeof(Real), cudaMemcpyDeviceToHost);
		
			// add red cell contributions to residual
			for (int i = 0; i < size_norm; ++i) {
				norm_L2 += bl_norm_L2[i];
			}
		#endif
			

		#if defined(TEXTURE) && defined(MEMOPT)
			black_kernel <<<dimGrid, dimBlock>>> (temp_red_d, temp_black_d, bl_norm_L2_d);
		#elif defined(TEXTURE) && !defined(MEMOPT)
			black_kernel <<<dimGrid, dimBlock>>> (temp_red_d, temp_red_d, bl_norm_L2_d);
		#elif !defined(TEXTURE) && defined(MEMOPT)
			black_kernel <<<dimGrid, dimBlock>>> (aP_d, aW_d, aE_d, aS_d, aN_d, b_d, temp_red_d, temp_black_d, bl_norm_L2_d);
		#else // neither defined
			black_kernel <<<dimGrid, dimBlock>>> (aP_d, aW_d, aE_d, aS_d, aN_d, b_d, temp_red_d, temp_red_d, bl_norm_L2_d);
		#endif
		
		// transfer residual value(s) back to CPU and 
		// add black cell contributions to residual
		#ifdef ATOMIC
			cudaMemcpy (bl_norm_L2, bl_norm_L2_d, sizeof(Real), cudaMemcpyDeviceToHost);
			norm_L2 = *bl_norm_L2;
		#else
			cudaMemcpy (bl_norm_L2, bl_norm_L2_d, size_norm * sizeof(Real), cudaMemcpyDeviceToHost);
			for (int i = 0; i < size_norm; ++i) {
				norm_L2 += bl_norm_L2[i];
			}
		#endif
		
		// calculate residual
		norm_L2 = sqrt(norm_L2 / ((Real)size));

		if (iter % 100 == 0) printf("%5d, %0.6f\n", iter, norm_L2);
		
		// if tolerance has been reached, end SOR iterations
		if (norm_L2 < tol) {
			break;
		}	
	}
	
	// transfer final temperature values back
	cudaMemcpy (temp_red, temp_red_d, size_temp * sizeof(Real), cudaMemcpyDeviceToHost);
	#ifdef MEMOPT
		cudaMemcpy (temp_black, temp_red_d, size_temp * sizeof(Real), cudaMemcpyDeviceToHost);
	#endif
	
	/////////////////////////////////
	// end timer
	//time = walltime(&time);
	//clock_t end_time = clock();
	double runtime = GetTimer();
	/////////////////////////////////
	
	printf("GPU\n");
	printf("Iterations: %i\n", iter);
	//printf("Time: %f\n", (end_time - start_time) / (double)CLOCKS_PER_SEC);
	printf("Total time: %f s\n", runtime / 1000.0);
	
	// print temperature data to file
	FILE * pfile;
	pfile = fopen("temp_gpu.dat", "w");
	
	if (pfile != NULL) {
		fprintf(pfile, "#x\ty\ttemp(K)\n");
		
		int row, col;
		for (row = 1; row < NUM + 1; ++row) {
			for (col = 1; col < NUM + 1; ++col) {
				Real x_pos = (col - 1) * dx + (dx / 2);
				Real y_pos = (row - 1) * dy + (dy / 2);
				
				if ((row + col) % 2 == 0) {
					// even, so red cell
					#ifdef MEMOPT
						int ind = col * num_rows + (row + (col % 2)) / 2;
					#else
						int ind = ((col - 1) * NUM ) + row - 1;
					#endif
					fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_red[ind]);
				} else {
					// odd, so black cell
					#ifdef MEMOPT
						int ind = col * num_rows + (row + ((col + 1) % 2)) / 2;
						fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_black[ind]);
					#else
						int ind = ((col - 1) * NUM ) + row - 1;
						fprintf(pfile, "%f\t%f\t%f\n", x_pos, y_pos, temp_red[ind]);
					#endif
				}	
			}
			fprintf(pfile, "\n");
		}
	}
	fclose(pfile);
	
	// free device memory
	cudaFree(aP_d);
	cudaFree(aW_d);
	cudaFree(aE_d);
	cudaFree(aS_d);
	cudaFree(aN_d);
	cudaFree(b_d);
	cudaFree(temp_red_d);
	#ifdef MEMOPT
		cudaFree(temp_black_d);
	#endif
	
	cudaFree(bl_norm_L2_d);
	
	#ifdef TEXTURE
		// unbind textures
		cudaUnbindTexture(aP_t);
		cudaUnbindTexture(aW_t);
		cudaUnbindTexture(aE_t);
		cudaUnbindTexture(aS_t);
		cudaUnbindTexture(aN_t);
		cudaUnbindTexture(b_t);
	#endif
	
	free(aP);
	free(aW);
	free(aE);
	free(aS);
	free(aN);
	free(b);
	free(temp_red);
	free(temp_black);
	free(bl_norm_L2);
	
	checkCudaErrors (cudaDeviceReset());
	
	return 0;
}
