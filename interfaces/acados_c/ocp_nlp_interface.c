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

#include "acados_c/ocp_nlp_interface.h"

//external
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "acados/ocp_nlp/ocp_nlp_cost_ls.h"
#include "acados/ocp_nlp/ocp_nlp_sqp.h"

ocp_nlp_solver_config *ocp_nlp_config_create(ocp_nlp_solver_plan plan, int N)
{
    int bytes = ocp_nlp_solver_config_calculate_size(N);
	void *config_mem = calloc(1, bytes);
	ocp_nlp_solver_config *config = ocp_nlp_solver_config_assign(N, config_mem);

    if (plan.nlp_solver == SQP_GN)
    {
        config->evaluate = &ocp_nlp_sqp;
        config->opts_calculate_size = &ocp_nlp_sqp_opts_calculate_size;
        config->opts_assign = &ocp_nlp_sqp_opts_assign;
        config->opts_initialize_default = &ocp_nlp_sqp_opts_initialize_default;
        config->memory_calculate_size = &ocp_nlp_sqp_memory_calculate_size;
        config->memory_assign = &ocp_nlp_sqp_memory_assign;
        config->workspace_calculate_size = &ocp_nlp_sqp_workspace_calculate_size;

        // QP solver
        config->qp_solver = ocp_qp_config_create(plan.ocp_qp_solver_plan);
        
        // LS cost
        for (int i = 0; i <= N; ++i)
        {
            ocp_nlp_cost_ls_config_initialize_default(config->cost[i]);
			// NOTE(giaf) it should be something like: (we may need to add the config_initialize_default in some modules)
            // plan->cost[i]->config_initialize_default(config->cost[i]);
        }

        // Dynamics
        for (int i = 0; i < N; ++i)
        {
		    ocp_nlp_dynamics_config_initialize_default(config->dynamics[i]);
		    config->dynamics[i]->sim_solver = sim_config_create(plan.sim_solver_plan[i]);
        }

        // Constraints
        for (int i = 0; i <= N; ++i)
		    ocp_nlp_constraints_linear_config_initialize_default(config->constraints[i]);
    }
    else
    {
        printf("Solver not available!\n");
        exit(1);
    }
    return config;
}



ocp_nlp_dims *ocp_nlp_dims_create(int N)
{
    int bytes = ocp_nlp_dims_calculate_size(N);

	void *ptr = calloc(1, bytes);

	ocp_nlp_dims *dims = ocp_nlp_dims_assign(N, ptr);

    return dims;
}



ocp_nlp_in *ocp_nlp_in_create(ocp_nlp_solver_config *config, ocp_nlp_dims *dims)
{
	int bytes = ocp_nlp_in_calculate_size(config, dims);

	void *ptr = calloc(1, bytes);

	ocp_nlp_in *nlp_in = ocp_nlp_in_assign(config, dims, ptr);

    return nlp_in;
}



ocp_nlp_out *ocp_nlp_out_create(ocp_nlp_solver_config *config, ocp_nlp_dims *dims)
{
	int bytes = ocp_nlp_out_calculate_size(config, dims);

	void *ptr = calloc(1, bytes);
	
    ocp_nlp_out *nlp_out = ocp_nlp_out_assign(config, dims, ptr);

    return nlp_out;
}



void *ocp_nlp_opts_create(ocp_nlp_solver_config *config, ocp_nlp_dims *dims)
{
    int bytes = config->opts_calculate_size(config, dims);

	void *ptr = calloc(1, bytes);

	void *opts = config->opts_assign(config, dims, ptr);

    config->opts_initialize_default(config, dims, opts);

    return opts;
}



int ocp_nlp_calculate_size(ocp_nlp_solver_config *config, ocp_nlp_dims *dims, void *opts_)
{
    int bytes = sizeof(ocp_nlp_solver);

    bytes += config->memory_calculate_size(config, dims, opts_);
    bytes += config->workspace_calculate_size(config, dims, opts_);

    return bytes;
}


ocp_nlp_solver *ocp_nlp_assign(ocp_nlp_solver_config *config, ocp_nlp_dims *dims, void *opts_, void *raw_memory)
{
    char *c_ptr = (char *) raw_memory;

    ocp_nlp_solver *solver = (ocp_nlp_solver *) c_ptr;
    c_ptr += sizeof(ocp_nlp_solver);

    solver->config = config;
    solver->dims = dims;
    solver->opts = opts_;

    solver->mem = config->memory_assign(config, dims, opts_, c_ptr);
    c_ptr += config->memory_calculate_size(config, dims, opts_);

    solver->work = (void *) c_ptr;
    c_ptr += config->workspace_calculate_size(config, dims, opts_);

    assert((char *) raw_memory + ocp_nlp_calculate_size(config, dims, opts_) == c_ptr);

    return solver;
}


ocp_nlp_solver *ocp_nlp_create(ocp_nlp_solver_config *config, ocp_nlp_dims *dims, void *opts_)
{
    int bytes = ocp_nlp_calculate_size(config, dims, opts_);

    void *ptr = calloc(1, bytes);

    ocp_nlp_solver *solver = ocp_nlp_assign(config, dims, opts_, ptr);

    return solver;
}



int ocp_nlp_solve(ocp_nlp_solver *solver, ocp_nlp_in *nlp_in, ocp_nlp_out *nlp_out)
{
    return solver->config->evaluate(solver->config, nlp_in->dims, nlp_in, nlp_out, solver->opts, solver->mem, solver->work);
}