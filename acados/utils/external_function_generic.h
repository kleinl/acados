/*
 *    This file is part of acados.
 *
 *    acados is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    acados is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with acados; if not, write to the Free Software Foundation,
 *    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#ifndef ACADOS_UTILS_EXTERNAL_FUNCTION_GENERIC_H_
#define ACADOS_UTILS_EXTERNAL_FUNCTION_GENERIC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/utils/types.h"



/************************************************
* generic external function
************************************************/

// prototype of an external function
typedef struct
{
	// public members (have to be before private ones)
	void (* evaluate) (void *, double *, double *);
	// private members
	// .....
} external_function_generic;



/************************************************
* casadi external function
************************************************/

typedef struct
{
	// public members (have to be the same as in the prototype, and before the private ones)
	void (* evaluate) (void *, double *, double *);
	// private members
	void *ptr_ext_mem; // pointer to external memory
	int (*casadi_fun) (const double **, double **, int *, double *, int);
	int (*casadi_work) (int *, int *, int *, int *);
	const int * (*casadi_sparsity_in) (int);
	const int * (*casadi_sparsity_out) (int);
	int (*casadi_n_in) ();
	int (*casadi_n_out) ();
	double **args;
	double **res;
	double *w;
	int *iw;
	int *args_size; // size of args[i]
	int *res_size; // size of res[i]
	int args_num; // number of args arrays
	int args_size_tot; // total size of args arrays
	int res_num; // number of res arrays
	int res_size_tot; // total size of res arrays
	int in_num; // number of input arrays
	int out_num; // number of output arrays
	int iw_size; // number of ints for worksapce
	int w_size; // number of dobules for workspace
} external_function_casadi;

//
int external_function_casadi_calculate_size(external_function_casadi *fun);
//
void external_function_casadi_assign(external_function_casadi *fun, void *mem);
//
void external_function_casadi_wrapper(void *self, double *in, double *out);



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_UTILS_EXTERNAL_FUNCTION_GENERIC_H_