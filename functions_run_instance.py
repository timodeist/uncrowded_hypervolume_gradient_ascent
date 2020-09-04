import numpy as np
import os
import pdb
import random

from class_optimizer import optimizer
from functions_optimization_benchmarks import load_benchmark

def run_instance(problem_name,optimizer_name,p,step_size_factor,rand_seed):
    # set seeds
    random.seed(rand_seed)
    np.random.seed(rand_seed)

    # get path to parent folder of function
    base_path = os.path.dirname(os.path.abspath(__file__))

    # define some properties
    n_mo_sol = p # number of mo-solutions
    n_parameters = 10 # number of parameters
    n_mo_obj = 2 # number of objectives needs to be fixed at 2
    n_iterations = 10**3 # number of optimizer iterations
    use_obj_space_normalization = True # whether or not to normalize uhv gradients in objective space

    # load benchmark problem
    fitness_func,fitness_grad,init_lb,init_ub,lb,ub,ref_point,paretoFrontFileName,problemNumber,f_params_list = load_benchmark(problem_name,n_parameters)

    # randomize initial mo-solutions
    init_values = np.zeros((n_parameters,n_mo_sol))
    for i_par in range(0,n_parameters):
        init_values[i_par,:] = np.random.uniform(init_lb[i_par],init_ub[i_par],(n_mo_sol))

    # create output file name
    output_file_name_prefix = optimizer_name + '_problem_' + problem_name + '_p_' + str(n_mo_sol) + '_run_' + str(rand_seed)
    output_file_name = os.path.join('statistics',output_file_name_prefix + '.dat')
    
    # initialize step size
    step_size = float(step_size_factor * np.mean(np.abs(init_ub-init_lb)))

    # instantiate optimizer
    uhv_opt = optimizer(n_mo_sol,n_parameters,step_size,use_obj_space_normalization,n_iterations,output_file_name,ref_point,init_values)

    # initialize mo_sol and mo_gradient
    mo_sol = init_values
    mo_obj_val = np.zeros((n_mo_obj,n_mo_sol))
    mo_gradient = np.zeros((n_mo_obj,n_mo_sol,n_parameters))
    
    #  optimizer iterations
    for i in range(0,n_iterations):
        # write iteration number to console
        if np.mod(i,np.floor(n_iterations/10)) == 0:
            print('Learning iteration: ' + str(i))

        # compute obj. func. values and gradients per objective
        for i_mo_sol in range(0,n_mo_sol):
            for i_obj in range(0,n_mo_obj):
                mo_obj_val[i_obj,i_mo_sol] = fitness_func[i_obj](mo_sol[:,i_mo_sol],f_params_list[i_obj])
                mo_gradient[i_obj,i_mo_sol,:] = fitness_grad[i_obj](mo_sol[:,i_mo_sol],f_params_list[i_obj])

        # use optimizer to update mo_sol
        mo_sol = uhv_opt.do_step(mo_sol, mo_obj_val, mo_gradient)