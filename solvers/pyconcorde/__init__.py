import os

try:
    from .concorde.tsp import TSPSolver
except:
    ori_dir = os.getcwd()
    os.chdir('solvers/pyconcorde')
    os.system("python ./setup.py build_ext --inplace")
    os.chdir(ori_dir)
    from .concorde.tsp import TSPSolver
    
    