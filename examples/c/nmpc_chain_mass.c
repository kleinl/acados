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
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_i_aux_ext_dep.h"

#include "acados_c/external_function_interface.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/sim_interface.h"


// TODO(dimitris): use only the strictly necessary includes here
// TODO(Lukas) : Standalone version
// TODO(Lukas) : one step sqp solver
// TODO(Lukas) : change modle in chain_mass.m:108


#include "acados/utils/mem.h"
#include "acados/utils/print.h"
#include "acados/utils/timing.h"
#include "acados/utils/types.h"

#include "acados/sim/sim_common.h"
#include "acados/ocp_nlp/ocp_nlp_sqp.h"
#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados/ocp_nlp/ocp_nlp_cost_common.h"
#include "acados/ocp_nlp/ocp_nlp_cost_ls.h"
#include "acados/ocp_nlp/ocp_nlp_cost_nls.h"
#include "acados/ocp_nlp/ocp_nlp_cost_external.h"
#include "acados/ocp_nlp/ocp_nlp_dynamics_cont.h"
#include "acados/ocp_nlp/ocp_nlp_dynamics_disc.h"
#include "acados/ocp_nlp/ocp_nlp_constraints_bgh.h"

#include "nmpc_chain_mass/chain_model_impl.h"

// xN
#include "nmpc_chain_mass/xN_nm2.c"
#include "nmpc_chain_mass/xN_nm3.c"
#include "nmpc_chain_mass/xN_nm4.c"
#include "nmpc_chain_mass/xN_nm5.c"
#include "nmpc_chain_mass/xN_nm6.c"

#define MAX_SQP_ITERS 1

#define NMF 2  // number of free masses: actually one more is used: possible values are 1,2,3,4,5

#define NU  3
#define NUM_STEPS   25        /* Number of real-time iterations. */
#define NRUNS 1				  /* Number of tries for the same problem. */

int NN = 0;


static void select_dynamics_casadi(int N, int num_free_masses,
	external_function_casadi *impl_ode_fun,
	external_function_casadi *impl_ode_fun_jac_x_xdot,
	external_function_casadi *impl_ode_fun_jac_x_xdot_u,
	external_function_casadi *impl_ode_jac_x_xdot_u)
{
	switch (num_free_masses)
	{
		case 1:
			for (int ii = 0; ii < N; ii++)
			{
				impl_ode_fun[ii].casadi_fun = &casadi_impl_ode_fun_chain_nm2;
				impl_ode_fun[ii].casadi_work = &casadi_impl_ode_fun_chain_nm2_work;
				impl_ode_fun[ii].casadi_sparsity_in = &casadi_impl_ode_fun_chain_nm2_sparsity_in;
				impl_ode_fun[ii].casadi_sparsity_out = &casadi_impl_ode_fun_chain_nm2_sparsity_out;
				impl_ode_fun[ii].casadi_n_in = &casadi_impl_ode_fun_chain_nm2_n_in;
				impl_ode_fun[ii].casadi_n_out = &casadi_impl_ode_fun_chain_nm2_n_out;

				impl_ode_fun_jac_x_xdot[ii].casadi_fun = &casadi_impl_ode_fun_jac_x_xdot_chain_nm2;
				impl_ode_fun_jac_x_xdot[ii].casadi_work = &casadi_impl_ode_fun_jac_x_xdot_chain_nm2_work;
				impl_ode_fun_jac_x_xdot[ii].casadi_sparsity_in = &casadi_impl_ode_fun_jac_x_xdot_chain_nm2_sparsity_in;
				impl_ode_fun_jac_x_xdot[ii].casadi_sparsity_out = &casadi_impl_ode_fun_jac_x_xdot_chain_nm2_sparsity_out;
				impl_ode_fun_jac_x_xdot[ii].casadi_n_in = &casadi_impl_ode_fun_jac_x_xdot_chain_nm2_n_in;
				impl_ode_fun_jac_x_xdot[ii].casadi_n_out = &casadi_impl_ode_fun_jac_x_xdot_chain_nm2_n_out;

				impl_ode_fun_jac_x_xdot_u[ii].casadi_fun = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm2;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_work = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm2_work;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_sparsity_in = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm2_sparsity_in;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_sparsity_out = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm2_sparsity_out;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_n_in = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm2_n_in;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_n_out = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm2_n_out;

				impl_ode_jac_x_xdot_u[ii].casadi_fun = &casadi_impl_ode_jac_x_xdot_u_chain_nm2;
				impl_ode_jac_x_xdot_u[ii].casadi_work = &casadi_impl_ode_jac_x_xdot_u_chain_nm2_work;
				impl_ode_jac_x_xdot_u[ii].casadi_sparsity_in = &casadi_impl_ode_jac_x_xdot_u_chain_nm2_sparsity_in;
				impl_ode_jac_x_xdot_u[ii].casadi_sparsity_out = &casadi_impl_ode_jac_x_xdot_u_chain_nm2_sparsity_out;
				impl_ode_jac_x_xdot_u[ii].casadi_n_in = &casadi_impl_ode_jac_x_xdot_u_chain_nm2_n_in;
				impl_ode_jac_x_xdot_u[ii].casadi_n_out = &casadi_impl_ode_jac_x_xdot_u_chain_nm2_n_out;
			}
			break;
		case 2:
			for (int ii = 0; ii < N; ii++)
			{
				impl_ode_fun[ii].casadi_fun = &casadi_impl_ode_fun_chain_nm3;
				impl_ode_fun[ii].casadi_work = &casadi_impl_ode_fun_chain_nm3_work;
				impl_ode_fun[ii].casadi_sparsity_in = &casadi_impl_ode_fun_chain_nm3_sparsity_in;
				impl_ode_fun[ii].casadi_sparsity_out = &casadi_impl_ode_fun_chain_nm3_sparsity_out;
				impl_ode_fun[ii].casadi_n_in = &casadi_impl_ode_fun_chain_nm3_n_in;
				impl_ode_fun[ii].casadi_n_out = &casadi_impl_ode_fun_chain_nm3_n_out;

				impl_ode_fun_jac_x_xdot[ii].casadi_fun = &casadi_impl_ode_fun_jac_x_xdot_chain_nm3;
				impl_ode_fun_jac_x_xdot[ii].casadi_work = &casadi_impl_ode_fun_jac_x_xdot_chain_nm3_work;
				impl_ode_fun_jac_x_xdot[ii].casadi_sparsity_in = &casadi_impl_ode_fun_jac_x_xdot_chain_nm3_sparsity_in;
				impl_ode_fun_jac_x_xdot[ii].casadi_sparsity_out = &casadi_impl_ode_fun_jac_x_xdot_chain_nm3_sparsity_out;
				impl_ode_fun_jac_x_xdot[ii].casadi_n_in = &casadi_impl_ode_fun_jac_x_xdot_chain_nm3_n_in;
				impl_ode_fun_jac_x_xdot[ii].casadi_n_out = &casadi_impl_ode_fun_jac_x_xdot_chain_nm3_n_out;

				impl_ode_fun_jac_x_xdot_u[ii].casadi_fun = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_work = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_work;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_sparsity_in = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_sparsity_in;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_sparsity_out = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_sparsity_out;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_n_in = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_n_in;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_n_out = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm3_n_out;

				impl_ode_jac_x_xdot_u[ii].casadi_fun = &casadi_impl_ode_jac_x_xdot_u_chain_nm3;
				impl_ode_jac_x_xdot_u[ii].casadi_work = &casadi_impl_ode_jac_x_xdot_u_chain_nm3_work;
				impl_ode_jac_x_xdot_u[ii].casadi_sparsity_in = &casadi_impl_ode_jac_x_xdot_u_chain_nm3_sparsity_in;
				impl_ode_jac_x_xdot_u[ii].casadi_sparsity_out = &casadi_impl_ode_jac_x_xdot_u_chain_nm3_sparsity_out;
				impl_ode_jac_x_xdot_u[ii].casadi_n_in = &casadi_impl_ode_jac_x_xdot_u_chain_nm3_n_in;
				impl_ode_jac_x_xdot_u[ii].casadi_n_out = &casadi_impl_ode_jac_x_xdot_u_chain_nm3_n_out;
			}
			break;
		case 3:
			for (int ii = 0; ii < N; ii++)
			{
				impl_ode_fun[ii].casadi_fun = &casadi_impl_ode_fun_chain_nm4;
				impl_ode_fun[ii].casadi_work = &casadi_impl_ode_fun_chain_nm4_work;
				impl_ode_fun[ii].casadi_sparsity_in = &casadi_impl_ode_fun_chain_nm4_sparsity_in;
				impl_ode_fun[ii].casadi_sparsity_out = &casadi_impl_ode_fun_chain_nm4_sparsity_out;
				impl_ode_fun[ii].casadi_n_in = &casadi_impl_ode_fun_chain_nm4_n_in;
				impl_ode_fun[ii].casadi_n_out = &casadi_impl_ode_fun_chain_nm4_n_out;

				impl_ode_fun_jac_x_xdot[ii].casadi_fun = &casadi_impl_ode_fun_jac_x_xdot_chain_nm4;
				impl_ode_fun_jac_x_xdot[ii].casadi_work = &casadi_impl_ode_fun_jac_x_xdot_chain_nm4_work;
				impl_ode_fun_jac_x_xdot[ii].casadi_sparsity_in = &casadi_impl_ode_fun_jac_x_xdot_chain_nm4_sparsity_in;
				impl_ode_fun_jac_x_xdot[ii].casadi_sparsity_out = &casadi_impl_ode_fun_jac_x_xdot_chain_nm4_sparsity_out;
				impl_ode_fun_jac_x_xdot[ii].casadi_n_in = &casadi_impl_ode_fun_jac_x_xdot_chain_nm4_n_in;
				impl_ode_fun_jac_x_xdot[ii].casadi_n_out = &casadi_impl_ode_fun_jac_x_xdot_chain_nm4_n_out;

				impl_ode_fun_jac_x_xdot_u[ii].casadi_fun = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm4;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_work = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm4_work;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_sparsity_in = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm4_sparsity_in;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_sparsity_out = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm4_sparsity_out;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_n_in = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm4_n_in;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_n_out = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm4_n_out;

				impl_ode_jac_x_xdot_u[ii].casadi_fun = &casadi_impl_ode_jac_x_xdot_u_chain_nm4;
				impl_ode_jac_x_xdot_u[ii].casadi_work = &casadi_impl_ode_jac_x_xdot_u_chain_nm4_work;
				impl_ode_jac_x_xdot_u[ii].casadi_sparsity_in = &casadi_impl_ode_jac_x_xdot_u_chain_nm4_sparsity_in;
				impl_ode_jac_x_xdot_u[ii].casadi_sparsity_out = &casadi_impl_ode_jac_x_xdot_u_chain_nm4_sparsity_out;
				impl_ode_jac_x_xdot_u[ii].casadi_n_in = &casadi_impl_ode_jac_x_xdot_u_chain_nm4_n_in;
				impl_ode_jac_x_xdot_u[ii].casadi_n_out = &casadi_impl_ode_jac_x_xdot_u_chain_nm4_n_out;
			}
			break;
		case 4:
			for (int ii = 0; ii < N; ii++)
			{
				impl_ode_fun[ii].casadi_fun = &casadi_impl_ode_fun_chain_nm5;
				impl_ode_fun[ii].casadi_work = &casadi_impl_ode_fun_chain_nm5_work;
				impl_ode_fun[ii].casadi_sparsity_in = &casadi_impl_ode_fun_chain_nm5_sparsity_in;
				impl_ode_fun[ii].casadi_sparsity_out = &casadi_impl_ode_fun_chain_nm5_sparsity_out;
				impl_ode_fun[ii].casadi_n_in = &casadi_impl_ode_fun_chain_nm5_n_in;
				impl_ode_fun[ii].casadi_n_out = &casadi_impl_ode_fun_chain_nm5_n_out;

				impl_ode_fun_jac_x_xdot[ii].casadi_fun = &casadi_impl_ode_fun_jac_x_xdot_chain_nm5;
				impl_ode_fun_jac_x_xdot[ii].casadi_work = &casadi_impl_ode_fun_jac_x_xdot_chain_nm5_work;
				impl_ode_fun_jac_x_xdot[ii].casadi_sparsity_in = &casadi_impl_ode_fun_jac_x_xdot_chain_nm5_sparsity_in;
				impl_ode_fun_jac_x_xdot[ii].casadi_sparsity_out = &casadi_impl_ode_fun_jac_x_xdot_chain_nm5_sparsity_out;
				impl_ode_fun_jac_x_xdot[ii].casadi_n_in = &casadi_impl_ode_fun_jac_x_xdot_chain_nm5_n_in;
				impl_ode_fun_jac_x_xdot[ii].casadi_n_out = &casadi_impl_ode_fun_jac_x_xdot_chain_nm5_n_out;

				impl_ode_fun_jac_x_xdot_u[ii].casadi_fun = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm5;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_work = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm5_work;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_sparsity_in = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm5_sparsity_in;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_sparsity_out = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm5_sparsity_out;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_n_in = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm5_n_in;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_n_out = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm5_n_out;

				impl_ode_jac_x_xdot_u[ii].casadi_fun = &casadi_impl_ode_jac_x_xdot_u_chain_nm5;
				impl_ode_jac_x_xdot_u[ii].casadi_work = &casadi_impl_ode_jac_x_xdot_u_chain_nm5_work;
				impl_ode_jac_x_xdot_u[ii].casadi_sparsity_in = &casadi_impl_ode_jac_x_xdot_u_chain_nm5_sparsity_in;
				impl_ode_jac_x_xdot_u[ii].casadi_sparsity_out = &casadi_impl_ode_jac_x_xdot_u_chain_nm5_sparsity_out;
				impl_ode_jac_x_xdot_u[ii].casadi_n_in = &casadi_impl_ode_jac_x_xdot_u_chain_nm5_n_in;
				impl_ode_jac_x_xdot_u[ii].casadi_n_out = &casadi_impl_ode_jac_x_xdot_u_chain_nm5_n_out;
			}
			break;
		case 5:
			for (int ii = 0; ii < N; ii++)
			{
				impl_ode_fun[ii].casadi_fun = &casadi_impl_ode_fun_chain_nm6;
				impl_ode_fun[ii].casadi_work = &casadi_impl_ode_fun_chain_nm6_work;
				impl_ode_fun[ii].casadi_sparsity_in = &casadi_impl_ode_fun_chain_nm6_sparsity_in;
				impl_ode_fun[ii].casadi_sparsity_out = &casadi_impl_ode_fun_chain_nm6_sparsity_out;
				impl_ode_fun[ii].casadi_n_in = &casadi_impl_ode_fun_chain_nm6_n_in;
				impl_ode_fun[ii].casadi_n_out = &casadi_impl_ode_fun_chain_nm6_n_out;

				impl_ode_fun_jac_x_xdot[ii].casadi_fun = &casadi_impl_ode_fun_jac_x_xdot_chain_nm6;
				impl_ode_fun_jac_x_xdot[ii].casadi_work = &casadi_impl_ode_fun_jac_x_xdot_chain_nm6_work;
				impl_ode_fun_jac_x_xdot[ii].casadi_sparsity_in = &casadi_impl_ode_fun_jac_x_xdot_chain_nm6_sparsity_in;
				impl_ode_fun_jac_x_xdot[ii].casadi_sparsity_out = &casadi_impl_ode_fun_jac_x_xdot_chain_nm6_sparsity_out;
				impl_ode_fun_jac_x_xdot[ii].casadi_n_in = &casadi_impl_ode_fun_jac_x_xdot_chain_nm6_n_in;
				impl_ode_fun_jac_x_xdot[ii].casadi_n_out = &casadi_impl_ode_fun_jac_x_xdot_chain_nm6_n_out;

				impl_ode_fun_jac_x_xdot_u[ii].casadi_fun = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm6;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_work = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm6_work;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_sparsity_in = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm6_sparsity_in;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_sparsity_out = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm6_sparsity_out;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_n_in = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm6_n_in;
				impl_ode_fun_jac_x_xdot_u[ii].casadi_n_out = &casadi_impl_ode_fun_jac_x_xdot_u_chain_nm6_n_out;

				impl_ode_jac_x_xdot_u[ii].casadi_fun = &casadi_impl_ode_jac_x_xdot_u_chain_nm6;
				impl_ode_jac_x_xdot_u[ii].casadi_work = &casadi_impl_ode_jac_x_xdot_u_chain_nm6_work;
				impl_ode_jac_x_xdot_u[ii].casadi_sparsity_in = &casadi_impl_ode_jac_x_xdot_u_chain_nm6_sparsity_in;
				impl_ode_jac_x_xdot_u[ii].casadi_sparsity_out = &casadi_impl_ode_jac_x_xdot_u_chain_nm6_sparsity_out;
				impl_ode_jac_x_xdot_u[ii].casadi_n_in = &casadi_impl_ode_jac_x_xdot_u_chain_nm6_n_in;
				impl_ode_jac_x_xdot_u[ii].casadi_n_out = &casadi_impl_ode_jac_x_xdot_u_chain_nm6_n_out;
			}
			break;
		default:
			printf("Problem size not available\n");
			exit(1);
			break;
	}
	return;
}

void read_final_state(const int nx, const int num_free_masses, double *xN)
{
	double *ptr;
    switch (num_free_masses)
    {
        case 1:
            ptr = xN_nm2;
            break;
        case 2:
            ptr = xN_nm3;
            break;
        case 3:
            ptr = xN_nm4;
            break;
        case 4:
            ptr = xN_nm5;
            break;
        case 5:
            ptr = xN_nm6;
            break;
        default:
            printf("\nwrong number of free masses\n");
			exit(1);
            break;
    }
    for (int i = 0; i < nx; i++)
		xN[i] = ptr[i];
}

/************************************************
* main
************************************************/

int main()
{
	double solution[10] = {0};
	int NX = (6*NMF) - 3;
	for(int horizon = 0; horizon < 1; horizon++) {
		// NN is the horizon in 10 steps.
		NN = 10*(horizon + 1);
		/************************************************
		* problem dimensions
		************************************************/
		int nx[NN+1];
		int nu[NN+1];
		int nbx[NN+1];
		int nbu[NN+1];
		int nb[NN+1];
		int ng[NN+1];
		int nh[NN+1];
		int nq[NN+1];
		int ns[NN+1];
		int ny[NN+1];
		int nz[NN+1];
		for (int j = 0; j < NN + 1; j++) {
			nx[j] = 0;
			nu[j] = 0;
			nbx[j] = 0;
			nbu[j] = 0;
			nb[j] = 0;
			ng[j] = 0;
			nh[j] = 0;
			nq[j] = 0;
			ns[j] = 0;
			ny[j] = 0;
			nz[j] = 0;
		}

		nx[0] = NX;
		nu[0] = NU;
		nbx[0] = nx[0];
		nbu[0] = nu[0];
		nb[0] = nbu[0]+nbx[0];
		ng[0] = 0;
		nh[0] = 0;
		ny[0] = nx[0]+nu[0];

		for (int i = 1; i < NN; i++)
		{
			nx[i] = NX;
			nu[i] = NU;
			nbx[i] = NMF;
			nbu[i] = NU;
			nb[i] = nbu[i]+nbx[i];
			ng[i] = 0;
			nh[i] = 0;
			ny[i] = nx[i]+nu[i];
		}

		nx[NN] = NX;
		nu[NN] = 0;
		nbx[NN] = NX;
		nbu[NN] = 0;
		nb[NN] = nbu[NN]+nbx[NN];
		ng[NN] = 0;
		nh[NN] = 0;
		ny[NN] = nx[NN]+nu[NN];
		/************************************************
		* plan + config
		************************************************/
		ocp_nlp_solver_plan *plan = ocp_nlp_plan_create(NN);

		// TODO(dimitris): not necessarily GN, depends on cost module
		plan->nlp_solver = SQP;

		// NOTE(dimitris): switching between different objectives on each stage to test everything
		for (int i = 0; i <= NN; i++)
		{
			plan->nlp_cost[i] = LINEAR_LS;
		}

		//plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
		// plan->ocp_qp_solver_plan.qp_solver = FULL_CONDENSING_HPIPM;
		plan->ocp_qp_solver_plan.qp_solver = FULL_CONDENSING_QPOASES;
		//plan->ocp_qp_solver_plan.qp_solver = FULL_CONDENSING_OOQP;
		// plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_OOQP;

		// NOTE(dimitris): switching between different integrators on each stage to test everything
		for (int i = 0; i < NN; i++)
		{
			plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
			plan->sim_solver_plan[i].sim_solver = IRK;
		}


		for (int i = 0; i <= NN; i++)
			plan->nlp_constraints[i] = BGH;

		// TODO(dimitris): fix minor memory leak here
		ocp_nlp_solver_config *config = ocp_nlp_config_create(*plan, NN);

		/************************************************
		* ocp_nlp_dims
		************************************************/

		ocp_nlp_dims *dims = ocp_nlp_dims_create(config);
		ocp_nlp_dims_initialize(config, nx, nu, ny, nbx, nbu, ng, nh, nq, ns, nz, dims);

		/************************************************
		* dynamics
		************************************************/

		// implicit
		external_function_casadi *impl_ode_fun = malloc(NN*sizeof(external_function_casadi));
		external_function_casadi *impl_ode_fun_jac_x_xdot = malloc(NN*sizeof(external_function_casadi));
		external_function_casadi *impl_ode_fun_jac_x_xdot_u = malloc(NN*sizeof(external_function_casadi));
		external_function_casadi *impl_ode_jac_x_xdot_u = malloc(NN*sizeof(external_function_casadi));

		select_dynamics_casadi(NN, NMF, impl_ode_fun, impl_ode_fun_jac_x_xdot, impl_ode_fun_jac_x_xdot_u, impl_ode_jac_x_xdot_u);

		// impl_ode
		external_function_casadi_create_array(NN, impl_ode_fun);
		//
		external_function_casadi_create_array(NN, impl_ode_fun_jac_x_xdot);
		//
		external_function_casadi_create_array(NN, impl_ode_fun_jac_x_xdot_u);
		//
		external_function_casadi_create_array(NN, impl_ode_jac_x_xdot_u);

		/************************************************
		* Simulator
		************************************************/
		sim_solver_plan plan_sim;

		plan_sim.sim_solver = IRK;

		sim_solver_config *config_sim = sim_config_create(plan_sim);

		void *dims_sim = sim_dims_create(config_sim);
		config_sim->set_nx(dims_sim, NX);
		config_sim->set_nu(dims_sim, NU);

		sim_rk_opts *opts_sim = sim_opts_create(config_sim, dims_sim);

		opts_sim->ns = 4; // number of stages in rk integrator
		opts_sim->num_steps = 5; // number of integration steps
		opts_sim->sens_adj = true;

		sim_in *in = sim_in_create(config_sim, dims_sim);
		sim_out *out = sim_out_create(config_sim, dims_sim);

		in->T = 0.2;

		sim_set_model(config_sim, in, "impl_ode_fun", &impl_ode_fun[0]);
		sim_set_model(config_sim, in, "impl_ode_fun_jac_x_xdot", &impl_ode_fun_jac_x_xdot[0]);
		sim_set_model(config_sim, in, "impl_ode_jac_x_xdot_u", &impl_ode_jac_x_xdot_u[0]);

		sim_solver *sim_solver = sim_create(config_sim, dims_sim, opts_sim);

		real_t minACADOtLog[NUM_STEPS];
		for(int runs = 0; runs < NRUNS; ++runs) {
			double *x_current = malloc(NX*sizeof(double));
			double *xref = malloc(NX*sizeof(double));
			read_final_state(NX, NMF, xref);
			read_final_state(NX, NMF, x_current);

			/* The "real-time iterations" loop. */
			for(int iter = 0; iter < NUM_STEPS; ++iter)
			{
				/************************************************
				* problem data
				************************************************/

				double wall_pos = -0.1;
				double UMAX = 1;

				double x_pos_inf = +1e4;
				double x_neg_inf = -1e4;

				double uref[3] = {0.0, 0.0, 0.0};

				// idxb0
				int *idxb0 = malloc(nb[0]*sizeof(int));

				for (int i = 0; i < nb[0]; i++) idxb0[i] = i;

				// idxb1
				int *idxb1 = malloc(nb[1]*sizeof(int));
				for (int i = 0; i < NU; i++) idxb1[i] = i;

				for (int i = 0; i < NMF; i++) idxb1[NU+i] = NU + 6*i + 1;

				// idxbN
				int *idxbN = malloc(nb[NN]*sizeof(int));
				for (int i = 0; i < nb[NN]; i++)
					idxbN[i] = i;

				// lb0, ub0
				double *lb0 = malloc((NX+NU)*sizeof(double));
				double *ub0 = malloc((NX+NU)*sizeof(double));

				for (int i = 0; i < NU; i++)
				{
					lb0[i] = -UMAX;
					ub0[i] = +UMAX;
				}

				for (int i = 0; i < NX; i++) {
					lb0[NU+i] = x_current[i];
				}
				for (int i = 0; i < NX; i++) {
					ub0[NU+i] = x_current[i];
				}

				// lb1, ub1
				double *lb1 = malloc((NMF+NU)*sizeof(double));
				double *ub1 = malloc((NMF+NU)*sizeof(double));

				for (int j = 0; j < NU; j++)
				{
					lb1[j] = -UMAX;  // umin
					ub1[j] = +UMAX;  // umax
				}
				for (int j = 0; j < NMF; j++)
				{
					lb1[NU+j] = wall_pos;  // wall position
					ub1[NU+j] = x_pos_inf;
				}

				// lbN, ubN
				double *lbN = malloc(NX*sizeof(double));
				double *ubN = malloc(NX*sizeof(double));

				for (int i = 0; i < NX; i++)
				{
					lbN[i] = x_neg_inf;
					ubN[i] = x_pos_inf;
				}


				/************************************************
				* nlp_in
				************************************************/

				ocp_nlp_in *nlp_in = ocp_nlp_in_create(config, dims);

				// sampling times
				for (int ii=0; ii<NN; ii++)
					nlp_in->Ts[ii] = 0.2;

				// output definition: y = [x; u]

				/* cost */
				ocp_nlp_cost_ls_model *stage_cost_ls;


				for (int i = 0; i <= NN; i++)
				{
					stage_cost_ls = (ocp_nlp_cost_ls_model *) nlp_in->cost[i];

					// Cyt
					blasfeo_dgese(nu[i]+nx[i], ny[i], 0.0, &stage_cost_ls->Cyt, 0, 0);
						for (int j = 0; j < nu[i]; j++)
					BLASFEO_DMATEL(&stage_cost_ls->Cyt, j, nx[i]+j) = 25.0;
						for (int j = 0; j < nx[i]; j++)
					BLASFEO_DMATEL(&stage_cost_ls->Cyt, nu[i]+j, j) = 1.0;

					// W
					blasfeo_dgese(ny[i], ny[i], 0.0, &stage_cost_ls->W, 0, 0);
					for (int j = 0; j < 3+(NMF-1)*3; j++)
						BLASFEO_DMATEL(&stage_cost_ls->W, j, j) = 25;
					for (int j = 3+(NMF-1)*3; j < 3+(NMF-1)*6; j++)
						BLASFEO_DMATEL(&stage_cost_ls->W, j, j) = 1;
					for (int j = 3+(NMF-1)*6; j < ny[i]; j++)
						BLASFEO_DMATEL(&stage_cost_ls->W, j, j) = 0.01;

					// y_ref
					blasfeo_pack_dvec(nx[i], xref, &stage_cost_ls->y_ref, 0);
					blasfeo_pack_dvec(nu[i], uref, &stage_cost_ls->y_ref, nx[i]);
				}

				/* dynamics */
				int set_fun_status;

				for (int i=0; i<NN; i++)
				{
					set_fun_status = nlp_set_model_in_stage(config, nlp_in, i, "impl_ode_fun", &impl_ode_fun[i]);
					if (set_fun_status != 0) exit(1);
					set_fun_status = nlp_set_model_in_stage(config, nlp_in, i, "impl_ode_fun_jac_x_xdot", &impl_ode_fun_jac_x_xdot[i]);
					if (set_fun_status != 0) exit(1);
					set_fun_status = nlp_set_model_in_stage(config, nlp_in, i, "impl_ode_jac_x_xdot_u", &impl_ode_jac_x_xdot_u[i]);
					if (set_fun_status != 0) exit(1);
				}


				/* constraints */
				ocp_nlp_constraints_bgh_model **constraints = (ocp_nlp_constraints_bgh_model **) nlp_in->constraints;
				ocp_nlp_constraints_bgh_dims **constraints_dims = (ocp_nlp_constraints_bgh_dims **) dims->constraints;

				// fist stage
				nlp_bounds_bgh_set(constraints_dims[0], constraints[0], "lb", lb0);
				nlp_bounds_bgh_set(constraints_dims[0], constraints[0], "ub", ub0);
				constraints[0]->idxb = idxb0;

				// other stages
				for (int i = 1; i < NN; i++)
				{
					nlp_bounds_bgh_set(constraints_dims[i], constraints[i], "lb", lb1);
					nlp_bounds_bgh_set(constraints_dims[i], constraints[i], "ub", ub1);
					constraints[i]->idxb = idxb1;
				}
				nlp_bounds_bgh_set(constraints_dims[NN], constraints[NN], "lb", lbN);
				nlp_bounds_bgh_set(constraints_dims[NN], constraints[NN], "ub", ubN);
				constraints[NN]->idxb = idxbN;

				/************************************************
				* sqp opts
				************************************************/

				void *nlp_opts = ocp_nlp_opts_create(config, dims);
				ocp_nlp_sqp_opts *sqp_opts = (ocp_nlp_sqp_opts *) nlp_opts;

				for (int i = 0; i < NN; ++i)
				{
					ocp_nlp_dynamics_cont_opts *dynamics_stage_opts = sqp_opts->dynamics[i];
					sim_rk_opts *sim_opts = dynamics_stage_opts->sim_solver;
					sim_opts->ns = 2;
					sim_opts->num_steps = 2;
					sim_opts->jac_reuse = true;
				}

				sqp_opts->maxIter = MAX_SQP_ITERS;
				sqp_opts->min_res_g = 1e-9;
				sqp_opts->min_res_b = 1e-9;
				sqp_opts->min_res_d = 1e-9;
				sqp_opts->min_res_m = 1e-9;

				/************************************************
				* ocp_nlp out
				************************************************/

				ocp_nlp_out *nlp_out = ocp_nlp_out_create(config, dims);

				ocp_nlp_solver *solver = ocp_nlp_create(config, dims, nlp_opts);

				/************************************************
				* sqp solve
				************************************************/

				int status;

				// warm start output initial guess of solution
				for (int i=0; i<=NN; i++)
				{
					blasfeo_pack_dvec(nu[i], uref, nlp_out->ux+i, 0);
					blasfeo_pack_dvec(nx[i], xref, nlp_out->ux+i, nu[i]);
				}

				acados_timer timer;
				acados_tic(&timer);

				// call nlp solver
				status = ocp_nlp_solve(solver, nlp_in, nlp_out);
				printf("%d \n", status);

				double time = acados_toc(&timer) * 1000;
				if (iter > 4) {
				  if (runs > 0) {
					if (minACADOtLog[iter] > time) minACADOtLog[iter] = time;
				  } else minACADOtLog[iter] = time;
				}

				//printf("%f\n", time);
				ocp_nlp_sqp_memory *solver_mem = (ocp_nlp_sqp_memory *) solver->mem;

				int sqp_iter = solver_mem->sqp_iter;

			    for (int k =0; k < 3; k++) {
			        printf("x[%d] = \n", k);
					blasfeo_print_tran_dvec(nx[k], nlp_out->ux+k, nu[k]);
			    	printf("u[%d] = \n", k);
					blasfeo_print_tran_dvec(nu[k], nlp_out->ux+k, 0);
			    }
			    printf("u[N-1] = \n");
				blasfeo_print_tran_dvec(nu[NN-1], nlp_out->ux+NN-1, 0);
			    printf("x[N] = \n");
				blasfeo_print_tran_dvec(nx[NN], nlp_out->ux+NN, nu[NN]);

				/* Apply the new control to the system, first NU components. */

				double controls[3];
				blasfeo_unpack_dvec(3, nlp_out->ux, 0, controls);

				// x
				for (int ii = 0; ii < NX; ii++)
					in->x[ii] = x_current[ii];

				// p
				for (int ii = 0;ii < NU; ii++)
					in->u[ii] = controls[ii];

				if (iter < 5) {
					in->u[0] = -1;
					in->u[0] = 1;
					in->u[0] = 1;
					minACADOtLog[iter] = 0;
				}

				sim_solve(sim_solver, in, out);

				for (int ii = 0; ii < NX; ii++)
					printf("%f ", x_current[ii]);
				printf("\n");
				for (int ii = 0; ii < NU; ii++)
					printf("%f ", controls[ii]);
				printf("\n");

				for (int ii = 0; ii < NX; ii++)
					x_current[ii] = out->xn[ii];

				free(nlp_opts);
				free(nlp_in);
				free(nlp_out);
				free(solver);


				free(lb0);
				free(ub0);
				free(lb1);
				free(ub1);
				free(lbN);
				free(ubN);
				free(idxb0);
				free(idxb1);
				free(idxbN);
			}
			free(x_current);
			free(xref);
		}
		double maximum = minACADOtLog[0];
		for (int c = 1; c < 25; c++) {
			if (minACADOtLog[c] > maximum)
				maximum  = minACADOtLog[c];
		}
		external_function_casadi_free(impl_ode_fun);
		external_function_casadi_free(impl_ode_fun_jac_x_xdot);
		external_function_casadi_free(impl_ode_fun_jac_x_xdot_u);
		external_function_casadi_free(impl_ode_jac_x_xdot_u);
		free(in);
		free(out);
		free(plan);
		free(config);
		free(dims);
		free(config_sim);
		free(dims_sim);
		free(impl_ode_fun);
		free(impl_ode_fun_jac_x_xdot);
		free(impl_ode_fun_jac_x_xdot_u);
		free(impl_ode_jac_x_xdot_u);

		solution[horizon] = maximum;
	}
	for (int i = 0; i < 10; ++i) {
		printf("%f\n", solution[i]);
	}
	return 0;
}
