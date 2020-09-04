import numpy as np
import pdb
# uncomment for using WFG benchmark
#import optproblems.wfg

def load_benchmark(benchmark_name,dim):
    f1_params_list = list()
    f2_params_list = list()
    f_params_list = list()
    if benchmark_name == 'bi-sphere-convex':
        fitness_func = [f1_genmed_convex, f2_genmed_convex]
        fitness_grad = [f1_genmed_convex_grad, f2_genmed_convex_grad]
        paretoFrontFileName = ''
        ref = np.array([11,11])
        init_lb = -2 * np.ones((dim))
        init_ub = 2 * np.ones((dim))
        lb = -1000 * np.ones((dim))
        ub = 1000 * np.ones((dim))
        # lb = -2 * np.ones((dim))
        # ub = 2 * np.ones((dim))
        problemNumber = 26 # this number is used to match with a benchmark labelling scheme in another codebase
        f_params_list.append(f1_params_list)
        f_params_list.append(f2_params_list)
    elif benchmark_name == 'bi-sphere-concave':
        fitness_func = [f1_genmed_concave, f2_genmed_concave]
        fitness_grad = [f1_genmed_concave_grad, f2_genmed_concave_grad]
        ref = np.array([11, 11])
        paretoFrontFileName = ''
        init_lb = -2 * np.ones((dim))
        init_ub = 2 * np.ones((dim))
        lb = -1000 * np.ones((dim))
        ub = 1000 * np.ones((dim))
        # lb = -2 * np.ones((dim))
        # ub = 2 * np.ones((dim))
        problemNumber = 31
        f_params_list.append(f1_params_list)
        f_params_list.append(f2_params_list)
    elif benchmark_name == 'sphere-rotElli':
        fitness_func = [f1_genmed_convex, f_rotElli]
        fitness_grad = [f1_genmed_convex_grad, f_rotElli_grad]
        init_lb = -2 * np.ones((dim))
        init_ub = 2 * np.ones((dim))
        lb = -1000 * np.ones((dim))
        ub = 1000 * np.ones((dim))
        # lb = -2 * np.ones((dim))
        # ub = 2 * np.ones((dim))
        ref = np.array([11, 11])
        paretoFrontFileName = ''
        problemNumber = 29
        
        # weight matrix
        W = np.eye(dim) 
        for i_dim in range(0,dim):
            W[i_dim,i_dim] = 10.0 **(-6.00 * (i_dim/(dim-1.00)))
        # construct many dimensional rotation matrix
        rotationDegrees = 45
        rotationRadians = np.deg2rad(rotationDegrees)
        R = np.eye(dim)
        for i in range(0,dim-1):
            for j in range(i+1,dim):
                tempMat = np.eye(dim)
                tempMat[i,i] = np.cos(rotationRadians)
                tempMat[i,j] = -np.sin(rotationRadians)
                tempMat[j,i] = np.sin(rotationRadians)
                tempMat[j,j] = np.cos(rotationRadians)
                R = np.matmul(tempMat,R)
        # displacement vector
        d = np.zeros(dim)
        d[0] = 1
        f2_params_list.append(R)
        f2_params_list.append(W)
        f2_params_list.append(d)    
        f_params_list.append(f1_params_list)
        f_params_list.append(f2_params_list)

    elif benchmark_name == 'scaledSphere-rosenbrock':
        fitness_func = [f1_genmed_convex_scaled, f_rosenbrock]
        fitness_grad = [f1_genmed_convex_scaled_grad, f_rosenbrock_grad]
        init_lb = 0 * np.ones((dim))
        init_ub = 2 * np.ones((dim))
        lb = -1000 * np.ones((dim))
        ub = 1000 * np.ones((dim))
        # lb = -2 * np.ones((dim))
        # ub = 2 * np.ones((dim))
        ref = np.array([11, 11])
        paretoFrontFileName = ''
        problemNumber = 30
        f_params_list.append(f1_params_list)
        f_params_list.append(f2_params_list)
    elif benchmark_name == 'sphere-rastrigin-weak':
        fitness_func = [f1_genmed_convex, f_rastrigin_weak]
        fitness_grad = [f1_genmed_convex_grad, f_rastrigin_weak_grad]
        init_lb = -2 * np.ones((dim))
        init_ub = 2 * np.ones((dim))
        lb = -1000 * np.ones((dim))
        ub = 1000 * np.ones((dim))
        # lb = -2 * np.ones((dim))
        # ub = 2 * np.ones((dim))
        ref = np.array([11, 11])
        paretoFrontFileName = ''
        problemNumber = 32
        f_params_list.append(f1_params_list)
        f_params_list.append(f2_params_list)
    elif benchmark_name == 'sphere-rastrigin-strong':
        fitness_func = [f1_genmed_convex, f_rastrigin_strong]
        fitness_grad = [f1_genmed_convex_grad, f_rastrigin_strong_grad]
        init_lb = -2 * np.ones((dim))
        init_ub = 2 * np.ones((dim))
        lb = -1000 * np.ones((dim))
        ub = 1000 * np.ones((dim))
        # lb = -2 * np.ones((dim))
        # ub = 2 * np.ones((dim))
        ref = np.array([11, 11])
        paretoFrontFileName = ''
        problemNumber = 33
        f_params_list.append(f1_params_list)
        f_params_list.append(f2_params_list)
    elif benchmark_name[0:3] == 'wfg':
        wfg_problem_number = int(benchmark_name[3])
        
        fitness_grad = [f_wfg_dummy_grad, f_wfg_dummy_grad]
        init_lb = 0 * np.ones((1,dim))
        init_ub = np.arange(1,dim+1) * 2
        lb = 0 * np.ones((1,dim))
        ub = np.arange(1,dim+1) * 2
        ref = np.array([11, 11])
        paretoFrontFileName = ''
        problemNumber = 50 + wfg_problem_number
        number_of_variables = dim
        number_of_positional_variables = 4
        
        # load problem class
        if wfg_problem_number == 1:
            problem_instance = optproblems.wfg.WFG1(2,number_of_variables,number_of_positional_variables)
        elif wfg_problem_number == 2:
            problem_instance = optproblems.wfg.WFG2(2,number_of_variables,number_of_positional_variables)
        elif wfg_problem_number == 3:
            problem_instance = optproblems.wfg.WFG3(2,number_of_variables,number_of_positional_variables)
        elif wfg_problem_number == 4:
            problem_instance = optproblems.wfg.WFG4(2,number_of_variables,number_of_positional_variables)
        elif wfg_problem_number == 5:
            problem_instance = optproblems.wfg.WFG5(2,number_of_variables,number_of_positional_variables)
        elif wfg_problem_number == 6:
            problem_instance = optproblems.wfg.WFG6(2,number_of_variables,number_of_positional_variables)
        elif wfg_problem_number == 7:
            problem_instance = optproblems.wfg.WFG7(2,number_of_variables,number_of_positional_variables)
        elif wfg_problem_number == 8:
            problem_instance = optproblems.wfg.WFG8(2,number_of_variables,number_of_positional_variables)
        elif wfg_problem_number == 9:
            problem_instance = optproblems.wfg.WFG9(2,number_of_variables,number_of_positional_variables)

        my_prob = optproblems.base.Problem(problem_instance,2)
        
        
        # initialize random parameters as individual class
        my_individual = optproblems.base.Individual(np.random.random(number_of_variables))
        
        # create wrapper
        fitness_func = f_wfg_wrapper(my_prob,my_individual)

        f_params_list.append(f1_params_list)
        f_params_list.append(f2_params_list)
    else:
        raise ValueError('Unknown benchmark name.')
    return(fitness_func,fitness_grad,init_lb,init_ub,lb,ub,ref,paretoFrontFileName,problemNumber,f_params_list)


######################################
################### Walking Fish Group (WFG)
######################################

class f_wfg_wrapper():

    def __init__(self,prob_instance,indiv_instance):
        self.prob_instance = prob_instance
        self.indiv_instance = indiv_instance
    
    def __call__(self,x):
        self.indiv_instance.phenome = x
        self.prob_instance.evaluate(self.indiv_instance)
        return(self.indiv_instance.objective_values)

def f_wfg_dummy_grad(x,f_params_list):
    number_of_variables = np.size(x)
    return(np.zeros(number_of_variables))

######################################
################### rastrigin - strong
######################################
def f_rastrigin_strong(x,f_params_list):
    n = len(x)
    d0 = 0.5
    A = 10.0
    y = A * n  + (x[0]-d0)**2 - A*np.cos(2*np.pi*(x[0]-d0)) + np.matmul(x[1:],x[1:]) - A * np.sum(np.cos(2*np.pi*x[1:]))  
    y = y/(A * n + d0*d0)
    return(y)

def f_rastrigin_strong_grad(x,f_params_list):
    n = len(x)
    d0 = 0.5
    A = 10.0
    grad = np.zeros(n)
    grad[0] = 2 * (x[0] - d0) + A  * np.sin(2 * np.pi * (x[0] - d0)) * 2 * np.pi        
    grad[1:] = 2*x[1:] + A * np.sin(2*np.pi*x[1:]) * 2 * np.pi
    grad = grad/(A*n+d0*d0)
    return(grad)

######################################
################### rastrigin - weak
######################################
def f_rastrigin_weak(x,f_params_list):
    d0 = 0.5
    A = 10.0
    y = A + (x[0]-d0)**2 - A*np.cos(2*np.pi*(x[0]-d0)) + np.matmul(x[1:],x[1:]) 
    y = y/(A+ d0*d0)
    return(y)

def f_rastrigin_weak_grad(x,f_params_list):
    dim = len(x)
    d0 = 0.5
    A = 10.0
    grad = np.zeros(dim)
    grad[0] = 2 * (x[0] - d0) + A  * np.sin(2 * np.pi * (x[0] - d0)) * 2 * np.pi
    grad[1:] = 2*x[1:]
    grad = grad/(A+d0*d0)
    return(grad)


######################################
################### rosenbrock
######################################

def f_rosenbrock(x,f_params_list):
    dim = len(x)
    shiftI = np.eye(dim,dim,1)
    
    shiftI = shiftI[0:-1,:] # remove row corresponding to x[-1]
    gamma = 100 * (np.matmul(shiftI,x) - x[0:-1]**2.0)**2.0 + (1-x[0:-1])**2.0
    y = 1.0/(dim-1) * np.sum(gamma)
    return(y)

def f_rosenbrock_grad(x,f_params_list):
    dim = len(x)
    grad = np.zeros(dim)
    grad[0] = 100 *(-4*x[1]*x[0] + 4* x[0]**3.00) - 2 + 2*x[0]
    grad[-1] = 200 * (x[-1] - x[-2]**2.00)
    
    # compute derivatives for all other x-entries
    xCur = x[1:-1] # current x-entry
    xPrev = x[0:-2] # previous x-entry
    xNext = x[2:] # next x-entry
    
    # x values in the middle have derivatives like x[0] (without the -1) plus derivatives like x[1] (see out1[0] and out2[-1])
    grad[1:-1] = -400 * xCur * (xNext - xCur**2.00) - 2.00 * (1 - xCur) + 200 * (xCur - xPrev**2.00)
    grad = 1.0/(dim-1) * grad
    return(grad)

######################################
################### rotated ellipsoid
######################################

def f_rotElli(x,f_params_list):
    R = f_params_list[0]
    W = f_params_list[1]
    d = f_params_list[2]

    tempVal = np.linalg.multi_dot([np.sqrt(W),R,x]) - np.matmul(np.sqrt(W),d)
    y = np.matmul(tempVal.T,tempVal)
    return(y)

def f_rotElli_grad(x,f_params_list):
    R = f_params_list[0]
    W = f_params_list[1]
    d = f_params_list[2]

    grad = 2 * np.matmul( np.transpose(np.matmul(np.sqrt(W),R)) , (np.matmul(R,x) - d) )
    return(grad)
    
######################################
################### genmed 
######################################

def f1_genmed(x,exponent):
    dim = len(x)
    center = np.zeros(dim)
    y = np.inner((x-center),(x-center))**(exponent/2.0)
    return(y)

def f2_genmed(x,exponent):
    dim = len(x)
    center = np.zeros(dim)
    center[0] = 1.0
    y = np.inner((x-center),(x-center))**(exponent/2.0)
    return(y)

def f1_genmed_grad(x,exponent):
    dim = len(x)
    center = np.zeros(dim)
    grad = exponent * ( x-center ) * np.inner((x-center),(x-center))**(exponent/2.0 - 1)
    return(grad)

def f2_genmed_grad(x,exponent):
    dim = len(x)
    center = np.zeros(dim)
    center[0] = 1.0
    grad = exponent * ( x-center ) * np.inner((x-center),(x-center))**(exponent/2.0 - 1)
    return(grad)

################### genmed convex

def f1_genmed_convex(x,f_params_list):
    exponent = 2.0
    y = f1_genmed(x,exponent)
    return(y)

def f2_genmed_convex(x,f_params_list):
    exponent = 2.0
    y = f2_genmed(x,exponent)
    return(y)

def f1_genmed_convex_grad(x,f_params_list):
    exponent = 2.0
    grad = f1_genmed_grad(x,exponent)
    return(grad)

def f2_genmed_convex_grad(x,f_params_list):
    exponent = 2.0
    grad = f2_genmed_grad(x,exponent)
    return(grad)

################### genmed concave 

def f1_genmed_concave(x,f_params_list):
    exponent = 0.5
    y = f1_genmed(x,exponent)
    return(y)

def f2_genmed_concave(x,f_params_list):
    exponent = 0.5
    y = f2_genmed(x,exponent)
    return(y)

def f1_genmed_concave_grad(x,f_params_list):
    exponent = 0.5
    grad = f1_genmed_grad(x,exponent)
    return(grad)

def f2_genmed_concave_grad(x,f_params_list):
    exponent = 0.5
    grad = f2_genmed_grad(x,exponent)
    return(grad)

################### genmed convex scaled

def f1_genmed_convex_scaled(x,f_params_list):
    dim = len(x)
    exponent = 2.0
    y = 1.0/dim * f1_genmed(x,exponent)
    return(y)

def f1_genmed_convex_scaled_grad(x,f_params_list):
    dim = len(x)
    exponent = 2.0
    grad = 1.0/dim * f1_genmed_grad(x,exponent)    
    return(grad)