import ctypes
import platform
import os


os_name = platform.system().lower()
if os_name == "windows":
    raise NotImplementedError("Temporarily not supported for Windows platform")
else:
    try:
        lib = ctypes.CDLL('search/c_mcts_solver/tsp_mcts_solver.so')
    except:
        ori_dir = os.getcwd()
        os.chdir("search/c_mcts_solver")
        os.system("make clean")
        os.system("make")
        os.chdir(ori_dir)
        lib = ctypes.CDLL('search/c_mcts_solver/tsp_mcts_solver.so')
    c_mcts_solver = lib.mcts_solver
    c_mcts_solver.argtypes = [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),   
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int                     
    ]
    c_mcts_solver.restype = ctypes.POINTER(ctypes.c_int)
