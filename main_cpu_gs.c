/** CPU Laplace solver using Gauss–Seidel with SOR solver
 * \file main_cpu_gs.c
 *
 * \author Kyle E. Niemeyer
 * \date 09/21/2012
 *
 * Solves Laplace's equation in 2D (e.g., heat conduction in a rectangular plate)
 * using the Gauss–Seidel with sucessive overrelaxation (SOR).
 * 
 * Boundary conditions:
 * T = 0 at x = 0, x = L, y = 0
 * T = TN at y = H
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/** Double precision */
#define Real double

typedef unsigned int uint;

/** SOR relaxation parameter */
const Real omega = 1.85;

/** Problem size along one side; total number of cells is this squared */
#define NUM 128

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

/** Function to perform Gauss–Seidel iteration
 * 
 * \param[in]			aP					array of self coefficients
 * \param[in]			aW					array of west neighbor coefficients
 * \param[in]			aE					array of east neighbor coefficients
 * \param[in]			aS					array of south neighbor coefficients
 * \param[in]			aN					array of north neighbor coefficients
 * \param[in]			b						right-hand side array
 * \param[inout]	temp_red		array of cell temperatures
 * \param[out]		norm_L2			variable holding summed residuals
 */
void gauss_seidel (const Real * aP, const Real * aW, const Real * aE, 
								 	 const Real * aS, const Real * aN, const Real * b,
									 Real * temp, Real * norm_L2)
{
	*norm_L2 = 0.0;
	
	for (uint col = 1; col < NUM + 1; ++col) {
		for (uint row = 1; row < NUM + 1; ++row) {
			
			uint ind_temp = col * (NUM + 2) + row;		// temperature index
			uint ind = (col - 1) * NUM + (row - 1);		// coefficient vector index
			
			Real res = b[ind] + (aW[ind] * temp[(col - 1) * (NUM + 2) + row]
												 + aE[ind] * temp[(col + 1) * (NUM + 2) + row]
												 + aS[ind] * temp[col * (NUM + 2) + (row - 1)]
												 + aN[ind] * temp[col * (NUM + 2) + (row + 1)]);
			
			Real temp_old = temp[ind_temp];
			temp[ind_temp] = temp_old * (1.0 - omega) + omega * (res / aP[ind]);
			
			res = temp[ind_temp] - temp_old;
			*norm_L2 += (res * res);
			
		} // end for row
	} // end for col
} // end gauss_seidel

///////////////////////////////////////////////////////////////////////////////

/** Main function that solves Laplace's equation in 2D (heat conduction in plate)
 * 
 * Contains iteration loop for Gauss-Seidel with SOR
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
	uint num_rows = NUM + 2;
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
	Real *aP, *aW, *aE, *aS, *aN, *b, *temp, *temp_old;
	
	// arrays of coefficients
	aP = (Real *) calloc (size, sizeof(Real));
	aW = (Real *) calloc (size, sizeof(Real));
	aE = (Real *) calloc (size, sizeof(Real));
	aS = (Real *) calloc (size, sizeof(Real));
	aN = (Real *) calloc (size, sizeof(Real));
	
	// RHS
	b = (Real *) calloc (size, sizeof(Real));
	
	// temperature arrays
	temp = (Real *) calloc (size_temp, sizeof(Real));
	temp_old = (Real *) calloc (size_temp, sizeof(Real));
	
	// set coefficients
	fill_coeffs (num_rows - 2, num_cols - 2, th_cond, dx, dy, width, TN, aP, aW, aE, aS, aN, b);
	
	for (uint i = 0; i < size_temp; ++i) {
		temp[i] = 0.0;
		temp_old[i] = 0.0;
	}
	
	//////////////////////////////
	// start timer
	//double time, start_time = 0.0;
	//time = walltime(&start_time);
	clock_t start_time = clock();
	//////////////////////////////
	
	// Gauss-Seidel with SOR iteration loop
	for (iter = 1; iter <= it_max; ++iter) {
		
		// residual variable
		Real norm_L2 = 0.0;
		
		// gauss-seidel iteration
		gauss_seidel (aP, aW, aE, aS, aN, b, temp, &norm_L2);
		
		// calculate residual
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
	
	printf("CPU\nIterations: %i\n", iter);
	printf("Time: %f\n", (end_time - start_time) / (double)CLOCKS_PER_SEC);
	
	// write temperature data to file
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
	
	// free memory
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
