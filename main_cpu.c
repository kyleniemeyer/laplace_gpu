#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <time.h>
//#include "walltime.h"

#define Real double

typedef unsigned int uint;

/** SOR relaxation parameter */
const Real omega = 1.85;

void fill_coeffs (uint rowmax, uint colmax, Real th_cond, Real dx, Real dy,
									Real width, Real TN, Real * aP, Real * aW, Real * aE, 
									Real * aS, Real * aN, Real * b) {
  
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

void red_kernel (uint rowmax, uint colmax, const Real * aP, const Real * aW, const Real * aE, 
								 const Real * aS, const Real * aN, const Real * b, Real * temp)
{
	for (uint col = 1; col < colmax - 1; ++col) {
		for (uint row = 1; row < rowmax - 1; ++row) {
			
			// only red cell if even
			if ((row + col) % 2 == 0) {
				uint ind = (col - 1) * (rowmax - 2) + (row - 1);
				uint ind2 = col * rowmax + row;
				
				Real res = b[ind] + (aW[ind] * temp[(col - 1) * rowmax + row]
													 + aE[ind] * temp[(col + 1) * rowmax + row]
													 + aS[ind] * temp[col * rowmax + (row - 1)]
													 + aN[ind] * temp[col * rowmax + (row + 1)]);
				
				temp[ind2] = temp[ind2] * (1.0 - omega) + omega * (res / aP[ind]);
			}	
				
		} // end for row
	} // end for col
	
}

void black_kernel (uint rowmax, uint colmax, const Real * aP, const Real * aW, const Real * aE, 
								   const Real * aS, const Real * aN, const Real * b, Real * temp)
{
	for (uint col = 1; col < colmax - 1; ++col) {
		for (uint row = 1; row < rowmax - 1; ++row) {
			
			// only black cell if odd
			if ((row + col) % 2 == 1) {
				uint ind = (col - 1) * (rowmax - 2) + (row - 1);
				uint ind2 = col * rowmax + row;
				
				Real res = b[ind] + (aW[ind] * temp[(col - 1) * rowmax + row]
													 + aE[ind] * temp[(col + 1) * rowmax + row]
													 + aS[ind] * temp[col * rowmax + (row - 1)]
													 + aN[ind] * temp[col * rowmax + (row + 1)]);
				
				temp[ind2] = temp[ind2] * (1.0 - omega) + omega * (res / aP[ind]);	
			}
				
		} // end for row
	} // end for col
	
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
	uint num_rows = 2048 + 2;
	uint num_cols = 2048 + 2;
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
	
	/*
	for (uint i = 0; i < size; ++i) {
		printf("%i  %f %f %f %f %f %f\n", i, aP[i], aW[i], aE[i], aS[i], aN[i], b[i]);
	}
	*/
	
	//////////////////////////////
	// start timer
	//double time, start_time = 0.0;
	//time = walltime(&start_time);
	clock_t start_time = clock();
	//////////////////////////////
	
	// iteration loop
	for (iter = 1; iter <= it_max; ++iter) {
		
		// update red cells
		red_kernel (num_rows, num_cols, aP, aW, aE, aS, aN, b, temp);
		
		// update black cells
		black_kernel (num_rows, num_rows, aP, aW, aE, aS, aN, b, temp);
		
		// check residual to see if done with iterations
		Real norm_L2 = 0.0;
		for (uint i = 0; i < size_temp; ++i) {
			norm_L2 += (temp[i] - temp_old[i]) * (temp[i] - temp_old[i]);
			temp_old[i] = temp[i];
		}
		norm_L2 = sqrt(norm_L2 / (size));
		
		// if tolerance has been reached, end SOR iterations
		if (norm_L2 < tol) {
			break;
		}	
	}
	
	/////////////////////////////////
	// end timer
	//time = walltime(&time);
	clock_t end_time = clock();
	/////////////////////////////////
	
	printf("Iterations: %i\n", iter);
	printf("Time: %f\n", (end_time - start_time) / (double)CLOCKS_PER_SEC);
	
	FILE * pfile;
	pfile = fopen("temp_cpu.dat", "w");
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
	
	return 0;
}
