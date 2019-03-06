
% Dialog with which the user selects the folder where the acados libs
% reside.
if(~ exist('acados_lib_path', 'var'))
    acados_path = uigetdir('', 'Please select folder with acados libraries for dSpace');
    acados_lib_path = fullfile(acados_path, 'lib');
    acados_include_path = fullfile(acados_path, 'include');
    blasfeo_include_path = fullfile(acados_path, 'include', 'blasfeo', 'include');
end

% Compile S function and link with acados
mex(['-I', acados_include_path], ...
    ['-I', blasfeo_include_path], ...
    ['-I', '.'], ...
    ['-I', '/Users/robin/acados/'], ...
    ['-L', acados_lib_path], ...
    ['-D', 'ACADOS_WITH_HPMPC'], ...
    ['-D', 'ACADOS_WITH_QPDUNES'], ...
    ['-D', 'ACADOS_WITH_OSQP'], ...
    ['-D', 'ACADOS_WITH_QPOASES'], ...
    '-lacados', ...
    '-lhpmpc', ...
    '-lhpipm', ...
    '-lblasfeo', ...
    '-lqpOASES_e', ...
    '-lqpdunes', ...
    'engine_nmpc.c', ...
    'engine_impl_dae_fun.c', ...
    'engine_impl_dae_jac_x_xdot_u_z.c', ...
    'engine_impl_dae_fun_jac_x_xdot_z.c', ...
    'engine_ls_cost.c', ...
    'engine_ls_cost_N.c');