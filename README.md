# Uncrowded Hypervolume (UHV) gradient ascent
This is a Python implementation of the Adam-based Uncrowded Hypervolume Gradient Ascent algorithm (UHV-Adam) described in:

Deist, T.M., Maree, S.C., Alderliesten, T., Bosman, P.A.N. (2020). Multi-objective Optimization by Uncrowded Hypervolume Gradient Ascent.

Link to pre-print: https://arxiv.org/abs/2007.04846  
The final authenticated version of the manuscript will be available in the conference proceedings of Parallel Problem Solving from Nature – PPSN XVI.

To run the algorithm on the four quadratic benchmarks described in the manuscript (Table 2):

1) Clone the repository

2) Create a subfolder named 'statistics' within the base folder

3) Run `python3 script_run_experiment.py` 

Statistics of each optimization run can then be found in the subfolder 'statistics'.
