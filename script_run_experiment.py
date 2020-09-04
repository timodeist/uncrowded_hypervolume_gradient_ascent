import pdb

from functions_run_instance import run_instance

# user inputs
problem_name_list = ['bi-sphere-convex','sphere-rotElli','bi-sphere-concave','scaledSphere-rosenbrock'] # ['bi-sphere-convex','sphere-rotElli','bi-sphere-concave','scaledSphere-rosenbrock']
optimizer_name_list = ['py_uhv_adam'] # ['py_uhv_adam']
p = 9 # number of solutions (named MO-solutions in the manuscript)
number_of_runs = 10
step_size_factor = 0.01
base_seed = 2020

# loop over problems, optimizers, runs
for i_prob, problem_name in enumerate(problem_name_list):
    for i_opt, optimizer_name in enumerate(optimizer_name_list):
        for i_run in range(0,number_of_runs):
            rand_seed = base_seed+i_run
            run_instance(problem_name,optimizer_name,p,step_size_factor,rand_seed)