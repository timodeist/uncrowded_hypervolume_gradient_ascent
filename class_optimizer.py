
import numpy as np
import csv
import pdb

class optimizer():

    def __init__(self,p,n,init_step_size,use_obj_space_normalization,max_iter,output_file_name,ref_point,init_mo_sol = None):
        self.output_file_name = output_file_name
        self.create_statistics_file()
       
        # initialize step size, expand to array (one step size for each mo_sol) if only a scalar was given
        self.init_step_size = init_step_size
        if type(init_step_size) == float:
            self.step_size = self.init_step_size * np.ones(p)
        elif len(init_step_size) == p:
            self.step_size = self.init_step_size
        else:
            raise ValueError('length of init_step_size needs to be either 1 or p')
        self.sync_error_tolerance = 10.0**-3
        # used to compute UD to points slightliy nudged across reference or domination boundary
        self.ud_eps = 10.0**-5 
       
        self.debug_mode = False # set True if more statistic should be stored in the instance    
        self.use_obj_space_normalization = use_obj_space_normalization
        self.max_iter = max_iter
        self.n_obj = 2
        self.n_parameters = n
        self.n_mo_sol = p 

        self.init_mo_sol = init_mo_sol
        self.mo_sol = init_mo_sol
        
        self.ref_point = ref_point

        # initialize arrays
        self.mo_obj_val = np.zeros((self.n_obj,self.n_mo_sol))
        self.obj_space_uhv_gradient = np.zeros((self.n_obj,self.n_mo_sol))
        self.hv_gradient = np.zeros((self.n_obj,self.n_mo_sol))
        self.ud_gradient = np.zeros((self.n_obj,self.n_mo_sol))
        self.normalized_obj_space_uhv_gradient = np.zeros((self.n_obj,self.n_mo_sol))
        
        self.par_space_uhv_gradient = np.zeros((self.n_parameters,self.n_mo_sol))
        self.obj_space_uhv_gradient = np.zeros((self.n_obj,self.n_mo_sol))
        self.mo_gradient = np.zeros((self.n_obj,self.n_mo_sol,self.n_parameters))
        self.search_direction = np.zeros((self.n_parameters,self.n_mo_sol))

        # initialize counters
        self.eval_count = 0
        self.grad_eval_count = 0
        self.iter_number = -1
        # initialize values
        self.best_uhv = -np.inf
        self.uhv = -np.inf
        self.best_iter = -np.inf
        
        # adam settings
        self.adam_eps = 10**(-16)
        self.adam_m = np.zeros((self.n_parameters,self.n_mo_sol))
        self.adam_v = np.zeros((self.n_parameters,self.n_mo_sol))
        self.adam_b_mean = 0.9
        self.adam_b_var = 0.999 
        self.adam_b_step = 0.99 

        # initialize lists for record-keeping
        self.hv_list = list()
        self.ud_list = list()
        self.uhv_list = list()
        self.mo_sol_list = list()
        self.mo_obj_val_list = list()
        self.mo_gradient_list = list()
        self.search_direction_list = list()
        self.par_space_uhv_gradient_list = list()
        self.obj_space_uhv_gradient_list = list()

    def do_step(self,mo_sol, mo_obj_val,gradient):
        # do UHV-Adam step

        # increase counter
        self.iter_number += 1
        # store arguments in instance
        self.mo_sol = mo_sol
        self.mo_gradient = gradient
        # store previous mo_obj_val to update step size
        self.previous_mo_obj_val = self.mo_obj_val
        self.mo_obj_val = mo_obj_val
        # increment evaluation counter; when mo_obj_val is computed externally, we assume 1 evaluation per mo-solution
        self.eval_count += self.n_mo_sol
        # determine which mo-solutions are non-dominated
        self.determine_non_dom_mo_sol()
        # determine which mo-solutions dominate the reference point
        self.determine_ref_dom_mo_sol()
        # compute statistics: HV, UD, UHV
        self.compute_statistics()
        # compute step sizes per MO-solution
        self.compute_step_size()
        ##compute search directions per MO-solution
        # compute uhv_gradient in parameter space, normalized in objective space
        self.compute_os_normalized_uhv_gradient()
        # apply Adam on normalized UHV gradient
        self.search_direction = self.compute_adam(self.par_space_uhv_gradient)      
        # record statistics and write to text file
        self.record_progress()
        # update all MO-solutions
        self.mo_sol = self.mo_sol - self.step_size * self.search_direction
        return(self.mo_sol)

    def compute_os_normalized_uhv_gradient(self):       
        
        # compute the uhv_gradient in obj space (dUHV/dY)
        self.compute_hv_gradient_2d()
        self.compute_ud_gradient_2d()
        self.obj_space_uhv_gradient = self.hv_gradient - self.ud_gradient
        # check that hv_gradient and ud_gradient are not non-zero at the same time
        assert not (np.any(np.bitwise_and(np.any(self.hv_gradient != 0,axis = 0),np.any(self.ud_gradient != 0,axis = 0))))
        # normalize the uhv_gradient in obj space (||dUHV/dY|| == 1)
        self.normalized_obj_space_uhv_gradient = np.zeros((2,self.n_mo_sol))
        for i_mo_sol in range(0,self.n_mo_sol):
            w = np.sqrt(np.sum(self.obj_space_uhv_gradient[:,i_mo_sol]**2.0))
            # if normalization is deactivated or the length of the gradient is close to 0,
            # then leave the search direction un-normalized
            if np.isclose(w,0) or (not self.use_obj_space_normalization):
                w = 1
            self.normalized_obj_space_uhv_gradient[:,i_mo_sol] = self.obj_space_uhv_gradient[:,i_mo_sol]/w
        # compute the 'adjusted' UHV gradient in parameter space (dUHV/dX) ('adjusted' because we normalized above)
        self.par_space_uhv_gradient = np.zeros((self.n_parameters,self.n_mo_sol))
        for i_mo_sol in range(0,self.n_mo_sol):
            # times -1 because we minimize the two objectives
            self.par_space_uhv_gradient[:,i_mo_sol] = -1 * (self.normalized_obj_space_uhv_gradient[0,i_mo_sol] * self.mo_gradient[0,i_mo_sol,:] + self.normalized_obj_space_uhv_gradient[1,i_mo_sol] * self.mo_gradient[1,i_mo_sol,:])

    def compute_adam(self,search_direction):
        self.adam_search_direction = np.zeros((self.n_parameters,self.n_mo_sol))
        for i_mo_sol in range(0,self.n_mo_sol):
            # update weighted average of current and past gradients
            self.adam_m[:,i_mo_sol] = self.adam_b_mean * self.adam_m[:,i_mo_sol] + (1-self.adam_b_mean) * search_direction[:,i_mo_sol]
            # update weighted average of current and past squared gradients
            self.adam_v[:,i_mo_sol] = self.adam_b_var * self.adam_v[:,i_mo_sol] + (1-self.adam_b_var) *  search_direction[:,i_mo_sol]**2.0
            # adjust averages
            m_adj = self.adam_m[:,i_mo_sol]/(1-self.adam_b_mean**(self.iter_number+1))
            v_adj = self.adam_v[:,i_mo_sol]/(1-self.adam_b_var**(self.iter_number+1))
            # rescale weighted average of gradients with weighted average of its square
            self.adam_search_direction[:,i_mo_sol] = m_adj/(np.sqrt(v_adj)+self.adam_eps)
            assert not np.any(np.isnan(self.adam_m))
            assert not np.any(np.isnan(self.adam_v))
            assert not np.any(np.isnan(self.adam_search_direction))
        return(self.adam_search_direction)

    def compute_step_size(self):
        if self.previous_uhv > self.uhv:
            self.step_size = self.adam_b_step * self.step_size
        assert not np.any(np.isnan(self.step_size))

##########################################################################
############## statistics
##########################################################################

    def create_statistics_file(self):
            # create table with same format as in Stef's tables (values that are not computed are replaced by -99)
            header_row = ['Iter', 'Evals', 'Current_UHV', 'Best_HV', 'Best_n_non_dom']
            with open(self.output_file_name,'w') as file_handle:
                file_writer = csv.writer(file_handle, dialect = 'excel-tab')
                # add header
                file_writer.writerow(header_row)

    def write_iteration_output(self):
        # write output per generation
        with open(self.output_file_name,'a') as file_handle:
            file_writer = csv.writer(file_handle, dialect = 'excel-tab')
            cur_row = [self.iter_number, self.eval_count, self.uhv, self.best_hv, self.best_n_non_dom]
            file_writer.writerow(cur_row)

    def record_best(self):
        if self.uhv >= self.best_uhv:
            self.best_uhv = float(self.uhv)
            self.best_hv = float(self.hv)
            self.best_ud = float(self.ud)
            self.best_non_dom_gd = self.non_dom_gd
            self.best_non_dom_igd = self.non_dom_igd
            self.best_n_non_dom = self.n_non_dom
            self.best_mo_obj_val = self.mo_obj_val.copy()
            self.best_mo_sol = self.mo_sol.copy()
            self.best_iter = self.iter_number

    def record_progress(self):
        # check whether new solution improves current best solution
        self.record_best()
        # only write results in selected iterations
        if (self.iter_number < 100) or ( (self.iter_number < 1000) and (np.mod(self.iter_number,10) == 0) ) or (np.mod(self.iter_number,1000) == 0) or (self.iter_number >= (self.max_iter-1)):# or (self.iterations_without_best_hv_update_counter >= (self.best_hv_no_improvement_iteration_limit-1)):
            self.write_iteration_output()
        # store stats at every iteration for debugging
        self.hv_list.append(self.hv)
        self.ud_list.append(self.ud)
        self.uhv_list.append(self.uhv)
        self.mo_obj_val_list.append(self.mo_obj_val.copy())
        # optionally, store even more
        if self.debug_mode:
            self.mo_sol_list.append(self.mo_sol.copy())
            self.search_direction_list.append(self.search_direction.copy())
            self.mo_gradient_list.append(self.mo_gradient.copy())
            self.par_space_uhv_gradient_list.append(self.par_space_uhv_gradient.copy())
            self.obj_space_uhv_gradient_list.append(self.obj_space_uhv_gradient.copy())


    def compute_statistics(self):
        # compute size of set
        self.n_non_dom = self.non_dom_mo_sol.shape[1]
        # compute GD of non_dominated solutions (not implemented)
        self.non_dom_gd = -99.0
        # compute IGD of non_dominated solutions (not implemented)
        self.non_dom_igd = -99.0
        # compute HV
        self.hv = self.compute_hv_2d()
        # compute UD
        self.ud, self.ud_contr = self.compute_ud_2d()
        # store previous uhv for step size update
        self.previous_uhv = self.uhv
        # compute UHV
        self.uhv = self.hv - self.ud
    
        assert type(self.hv) == float
        assert type(self.ud) == float
        assert type(self.uhv) == float
        assert type(self.non_dom_gd) == float
        assert type(self.non_dom_igd) == float
        assert type(self.n_non_dom) == int

##########################################################################
############## HV, UD, HV gradient, UD gradient, etc.
##########################################################################

    def determine_non_dom_mo_sol(self):
        # get set of non-dominated solutions
        domination_rank = self.fast_non_dominated_sort(self.mo_obj_val)
        self.non_dom_indices = np.where(domination_rank == 0)
        self.non_dom_indices = self.non_dom_indices[0] # somehow this is necessary
        self.non_dom_mo_sol = self.mo_sol[:,self.non_dom_indices]
        self.non_dom_mo_obj_val = self.mo_obj_val[:,self.non_dom_indices]
        self.mo_sol_is_non_dominated = np.zeros(self.n_mo_sol,dtype = bool)
        self.mo_sol_is_non_dominated[self.non_dom_indices] = True
        self.mo_sol_is_dominated = np.bitwise_not(self.mo_sol_is_non_dominated)

    def determine_ref_dom_mo_sol(self):
        # select only mo-solutions that dominate the reference point

        ref_point_temp = self.ref_point[:,None] # add axis so that comparison works
        ref_dom_booleans = np.all(self.mo_obj_val < ref_point_temp  , axis = 0)
        ref_dom_indices = np.where(ref_dom_booleans == True)

        self.mo_sol_dominates_ref_point = ref_dom_booleans
        self.ref_dom_indices = ref_dom_indices[0] # somehow this is necessary
        self.ref_dom_mo_sol = self.mo_sol[:,self.ref_dom_indices]
        self.ref_dom_mo_obj_val = self.mo_obj_val[:,self.ref_dom_indices]
        self.n_ref_dom_mo_sol = self.ref_dom_mo_obj_val.shape[1]

        assert self.n_ref_dom_mo_sol >= 0
        assert not np.any(np.isnan(self.ref_dom_mo_sol))
        assert not np.any(np.isnan(self.ref_dom_mo_obj_val))

    def determine_ref_dom_non_dom_mo_sol(self):
        # select mo-solutions that dominate the reference point AND are non-dominated, sort in increasing order of objective zero

        ref_dom_non_dom_indices = np.intersect1d(self.ref_dom_indices,self.non_dom_indices)
        ref_dom_non_dom_obj_val = self.mo_obj_val[:,ref_dom_non_dom_indices]
        ref_dom_non_dom_mo_sol = self.mo_obj_val[:,ref_dom_non_dom_indices]
        n_ref_dom_non_dom_mo_sol = ref_dom_non_dom_obj_val.shape[1]

        # sort points in increasing order of objective one
        sort_indices = np.argsort(ref_dom_non_dom_obj_val[0,:])
        # sort_indices = sort_indices[0] # somehow this is necessary
        sorted_ref_dom_non_dom_obj_val = ref_dom_non_dom_obj_val[:,sort_indices]
        # use argsort to find indices that revert the previous sorting. Note to self: sketch an example
        inv_sort_indices = np.argsort(sort_indices)

        # assert that the indexing logic is correct (the inversion of the sorting)
        assert np.all( self.mo_obj_val[:,ref_dom_non_dom_indices] == sorted_ref_dom_non_dom_obj_val[:,inv_sort_indices])
        return(ref_dom_non_dom_mo_sol,ref_dom_non_dom_obj_val,ref_dom_non_dom_indices,n_ref_dom_non_dom_mo_sol,inv_sort_indices,sorted_ref_dom_non_dom_obj_val)

    def compute_hv_gradient_2d(self):
        # the hv gradient of given mo-solution is
        # for objective 0 (x-axis), the vertical length of the rectangle from the neighboring mo solution on the left to the given mo-solution.
        # for objective 1 (y-axis), the horizontal lenght of the rectangle from the neighboring mo solution on the right to the given mo-solution.
        # first and last mo-solutions need to consider the intersection point with the reference box as 'neighboring mo-solutions' (draw it, and then you see it)
        # if there is only one mo-solution which dominates the reference point and is non-dominated by any other mo-solution, consider the intersection points with the reference box as 'neighboring mo-solutions' (draw it, and then you see it)

        assert self.n_obj == 2
        # if no point dominates the reference point, return 0 for all gradients
        hv_gradient = np.zeros_like(self.mo_obj_val)
        if not self.n_ref_dom_mo_sol == 0:
            # select mo-solutions that dominate the reference point AND are non-dominated, sort in increasing order of objective zero
            _,_,ref_dom_non_dom_indices,n_ref_dom_non_dom_mo_sol,inv_sort_indices,sorted_ref_dom_non_dom_obj_val = self.determine_ref_dom_non_dom_mo_sol()            

            hv_gradient_sorted_ref_dom_non_dom = np.zeros((self.n_obj,n_ref_dom_non_dom_mo_sol))
            # if there is only one mo-solution that dominates the ref point and is non-dominated by other mo-solutions, the hv gradient is defined by the rectangle between the mo-solution and the reference point
            if n_ref_dom_non_dom_mo_sol == 1:
                hv_gradient_sorted_ref_dom_non_dom[0,0] = - ( self.ref_point[1] - sorted_ref_dom_non_dom_obj_val[1,0] )
                hv_gradient_sorted_ref_dom_non_dom[1,0] = - ( self.ref_point[0] - sorted_ref_dom_non_dom_obj_val[0,0] )
            elif n_ref_dom_non_dom_mo_sol > 1:
                # first mo-solution
                hv_gradient_sorted_ref_dom_non_dom[0,0] = - ( self.ref_point[1] - sorted_ref_dom_non_dom_obj_val[1,0] )
                hv_gradient_sorted_ref_dom_non_dom[1,0] = - ( sorted_ref_dom_non_dom_obj_val[0,(0+1)] - sorted_ref_dom_non_dom_obj_val[0,0] )
                # intermediate mo-solutions
                for i_mo_obj_val in range(1,n_ref_dom_non_dom_mo_sol-1): # 1 -1 because the first and last mo-solutions need to be treated separately
                    hv_gradient_sorted_ref_dom_non_dom[0,i_mo_obj_val] = - ( sorted_ref_dom_non_dom_obj_val[1,(i_mo_obj_val-1)] - sorted_ref_dom_non_dom_obj_val[1,i_mo_obj_val] )
                    hv_gradient_sorted_ref_dom_non_dom[1,i_mo_obj_val] = - ( sorted_ref_dom_non_dom_obj_val[0,(i_mo_obj_val+1)] - sorted_ref_dom_non_dom_obj_val[0,i_mo_obj_val] )

                # last last mo-solution
                hv_gradient_sorted_ref_dom_non_dom[0,-1] = - ( sorted_ref_dom_non_dom_obj_val[1,-2] - sorted_ref_dom_non_dom_obj_val[1,-1] )
                hv_gradient_sorted_ref_dom_non_dom[1,-1] = - ( self.ref_point[0] - sorted_ref_dom_non_dom_obj_val[0,-1] )
            else:
                raise ValueError('Unknown case. There should always be 1 mo-solution in this if-statement.')
            
            hv_gradient[:,ref_dom_non_dom_indices] = hv_gradient_sorted_ref_dom_non_dom[:,inv_sort_indices]
        assert np.all(hv_gradient <= 0) # we are minimizing the mo-objectives. Therefore, an increase in the objectives should always yield to a decrease in HV.
        assert not np.any(np.isnan(hv_gradient))
        self.hv_gradient = hv_gradient



    def compute_hv_2d(self):
        # the HV for two objectives is a sum the area of rectangles 
        # each rectangle (except the rightmost one) has a non-dominated mo-solution as the bottom left corner point, the bottom right corner point has the y-coordinate of the next mo-solution. the top corner points have the y-coordinate of the reference point.
        # the last rectangle uses the coordinates of reference point and the last mo-solution to determine its corner points
        # the HV is computed iterating over the rectangles from left to right
        assert self.n_obj == 2
        # if no point dominates the reference point, return 0
        if self.n_ref_dom_mo_sol == 0:
            hv = np.zeros(1)
        else:
            # select mo-solutions that dominate the reference point AND are non-dominated, sort in increasing order of objective zero
            _,_,ref_dom_non_dom_indices,n_ref_dom_non_dom_mo_sol,inv_sort_indices,sorted_ref_dom_non_dom_obj_val = self.determine_ref_dom_non_dom_mo_sol()
            hv = np.zeros(1)
            # iteratively add the rectangle areas
            for i_mo_obj_val in range(0,n_ref_dom_non_dom_mo_sol-1): # -1 because the last rectangle needs to be treated separately
                hv += ( self.ref_point[1] - sorted_ref_dom_non_dom_obj_val[1,i_mo_obj_val] )  * ( sorted_ref_dom_non_dom_obj_val[0,(i_mo_obj_val+1)] - sorted_ref_dom_non_dom_obj_val[0,i_mo_obj_val] )
            # last rectangle
            hv += ( self.ref_point[1] - sorted_ref_dom_non_dom_obj_val[1,-1] ) * ( self.ref_point[0] - sorted_ref_dom_non_dom_obj_val[0,-1] )
            # note: in the case that there is only one solution, the for loop is not used and the last rectangle is also the only rectangle
        
        hv = float(hv)
        assert not np.isnan(hv)
        assert hv >= 0
        return(hv)

    def compute_ud_gradient_2d(self):
        # the UD gradient can be computed from the vertical and horizontal components of the UD (computed in compute_ud_2d)
        assert self.n_obj == 2
        self.ud_gradient = 2 * self.obj_space_ud_components
        assert not np.any(np.isnan(self.ud_gradient))

    def compute_ud_2d(self):
        # NOTE: this can be made more efficient by not computing distance of ALL 'inverse' corner points to a dominated point
        assert self.n_obj == 2
        self.obj_space_ud_components = np.zeros((2,self.n_mo_sol))
        # initialization: the closest point of an mo-solution is set to itself (the closest point is only recorded for possible future use)
        self.ud_closest_point = self.mo_obj_val.copy()

        # special case: if all solutions are non-dominated and all dominate the reference point, UD = 0
        if (self.n_non_dom == self.n_mo_sol) and (self.n_ref_dom_mo_sol == self.n_mo_sol):
            ud_contr = np.zeros(self.n_mo_sol)
            ud = np.zeros(1)
            ud = float(ud)
            return(ud,ud_contr)

        ## determine whether the UD is computed w.r.t. the reference box or the domination boundary of the non-dominated mo-solutions
        # if no point dominates the reference point, compute each mo_sol's distance to the reference box, then the reference point is the corner point (also called 'inverse corner point' or in code)
        # inverse corner points are the top right corner points of rectangles to which distances are computed
        ref_point_is_corner_point = False
        if self.n_ref_dom_mo_sol == 0:
            inv_corner_obj_val = self.ref_point
            n_inv_corner = 1
            ref_point_is_corner_point = True
            # add 2nd dimension to array because later we will iterate over all inv. corners via the 2nd dimension
            inv_corner_obj_val = inv_corner_obj_val[:,None] 
        else:
            # select mo-solutions that dominate the reference point AND are non-dominated, sort in increasing order of objective zero
            _,_,ref_dom_non_dom_indices,n_ref_dom_non_dom_mo_sol,inv_sort_indices,sorted_ref_dom_non_dom_obj_val = self.determine_ref_dom_non_dom_mo_sol()
            # if there is just one mo-solution on the domination boundary, it itself is the only cornerpoint
            if n_ref_dom_non_dom_mo_sol == 1:
                inv_corner_obj_val = sorted_ref_dom_non_dom_obj_val
                n_inv_corner = 1
            # otherwise construct the 'inverse' corner points from multiple mo-solutions on the domination boundary
            elif n_ref_dom_non_dom_mo_sol > 1:
                n_inv_corner = n_ref_dom_non_dom_mo_sol-1
                inv_corner_obj_val = np.zeros((self.n_obj,n_inv_corner))
                
                inv_corner_obj_val[0,:] = sorted_ref_dom_non_dom_obj_val[0,1:]
                inv_corner_obj_val[1,:] = sorted_ref_dom_non_dom_obj_val[1,:-1]
            else:
                raise ValueError('Unknown case. n_ref_dom_mo_sol != 0  but n_ref_dom_non_dom_mo_sol < 1.')

        ## compute UD
        # initialize ud contribution
        ud_contr = np.zeros(self.n_mo_sol)
        for i_mo_sol in range(0,self.n_mo_sol):
            # if solution is dominated or outside the reference box (i.e. not dominating the ref point), then compute its ud contribution, otherwise leave it at 0
            if self.mo_sol_is_dominated[i_mo_sol] or (not self.mo_sol_dominates_ref_point[i_mo_sol]):    
                dist_to_dom_boundary_list = list()
                horz_dist_list = list()
                vert_dist_list = list()
                # compute the distances to each of the rectangles/boxes 'underneath' the domination boundary
                for i_inv_cor in range(0,n_inv_corner):
                    dist_to_box, horz_dist, vert_dist = self.compute_eucl_distance_to_box(inv_corner_obj_val[:,i_inv_cor],self.mo_obj_val[:,i_mo_sol])
                    dist_to_dom_boundary_list.append(dist_to_box)
                    horz_dist_list.append(horz_dist)
                    vert_dist_list.append(vert_dist)
                # select the distance to the closest box and make their the domianted mo solution's UD contribution
                min_ind = np.argmin(dist_to_dom_boundary_list)
                ud_contr[i_mo_sol] = dist_to_dom_boundary_list[min_ind]
                # record the horizontal and vertical components for UD-gradient computation      
                self.obj_space_ud_components[0,i_mo_sol] = horz_dist_list[min_ind]
                self.obj_space_ud_components[1,i_mo_sol] = vert_dist_list[min_ind]
                # record the closest point (maybe useful in the future)
                self.ud_closest_point[:,i_mo_sol] = inv_corner_obj_val[:,min_ind]

        # the UD is the mean of the distances to the power of self.n_obj (for 2d: self.n_obj == 2)
        ud = np.mean(ud_contr**2.0)
        ud = float(ud)
        assert not np.isnan(ud)
        assert not np.any(np.isnan(ud_contr))
        assert ud >= 0
        assert np.all(ud_contr >=0)
        return(ud,ud_contr)


    def compute_eucl_distance_to_box(self,corner_point,free_point):
        horz_dist = free_point[0] - (corner_point[0] - self.ud_eps)
        vert_dist = free_point[1] - (corner_point[1] - self.ud_eps)

        # if the vertical distance is negative, the free point is below the corner point
        if (horz_dist > 0) and (vert_dist < 0):
            eucl_dist_to_box = horz_dist
            vert_dist = 0
        # if the horizontal distance is negative, the free point is to the left of the corner point
        elif (horz_dist < 0) and (vert_dist > 0):
            eucl_dist_to_box = vert_dist
            horz_dist = 0
        # if both distances are positive, the free point is to the top right of the corner point
        elif (horz_dist > 0) and (vert_dist > 0):
            # pythagoras
            eucl_dist_to_box = np.sqrt((horz_dist**2.0 + vert_dist**2.0))
        else:
            pdb.set_trace()
            raise ValueError('Unexpected case. Dominated point seems to dominate a corner point on the domination boundary or the reference point?')
        
        return(eucl_dist_to_box,horz_dist,vert_dist)

    def fast_non_dominated_sort(self,objVal):
        # As in Deb et al. (2002) NSGA-II
        N_OBJECTIVES = objVal.shape[0] 
        N_SOLUTIONS = objVal.shape[1]

        rankIndArray = - 999 * np.ones(N_SOLUTIONS, dtype = int) # -999 indicates unassigned rank
        solIndices = np.arange(0,N_SOLUTIONS) # array of 0 1 2 ... N_SOLUTIONS
        ## compute the entire domination matrix
        # dominationMatrix: (i,j) is True if solution i dominates solution j
        dominationMatrix = np.zeros((N_SOLUTIONS,N_SOLUTIONS), dtype = bool)
        for p in solIndices:
            objValA = objVal[:,p][:,None] # add [:,None] to preserve dimensions
            # objValArray =  np.delete(objVal, obj = p axis = 1) # dont delete solution p because it messes up indices
            dominates = self.check_domination(objValA,objVal)
            dominationMatrix[p,:] = dominates

        # count the number of times a solution is dominated
        dominationCounter = np.sum(dominationMatrix, axis = 0)

        ## find rank 0 solutions to initialize loop
        isRankZero = (dominationCounter == 0) # column and row binary indices of solutions that are rank 0
        # pdb.set_trace()
        rankZeroRowInd = solIndices[isRankZero] 
        # mark rank 0's solutions by -99 so that they are not considered as members of next rank
        dominationCounter[rankZeroRowInd] = -99
        # initialize rank counter at 0
        rankCounter = 0
        # assign solutions in rank 0 rankIndArray = 0
        rankIndArray[isRankZero] = rankCounter

        isInCurRank = isRankZero
        # while the current rank is not empty
        while not (np.sum(isInCurRank) == 0):
            curRankRowInd = solIndices[isInCurRank] # column and row numbers of solutions that are in current rank 
            # for each solution in current rank
            for p in curRankRowInd:
                # decrease domination counter of each solution dominated by solution p which is in the current rank
                dominationCounter[dominationMatrix[p,:]] -= 1 #dominationMatrix[p,:] contains indices of the solutions dominated by p
            # all solutions that now have dominationCounter == 0, are in the next rank		
            isInNextRank = (dominationCounter == 0)
            rankIndArray[isInNextRank] = rankCounter + 1	
            # mark next rank's solutions by -99 so that they are not considered as members of future ranks
            dominationCounter[isInNextRank] = -99
            # increase front counter
            rankCounter += 1
            # check which solutions are in current rank (next rank became current rank)
            isInCurRank = (rankIndArray == rankCounter)
            # if not np.all(isInNextRank == isInCurRank): # DEBUGGING, if it works fine, replace above assignment
                # pdb.set_trace()
        return(rankIndArray)

    def check_domination(self,obj_val_A,obj_val_array):
        dominates = ( np.any(obj_val_A < obj_val_array, axis = 0) & np.all(obj_val_A <= obj_val_array , axis = 0) )
        return(dominates)
