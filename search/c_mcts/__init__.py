import ctypes
import platform
import os


os_name = platform.system().lower()
if os_name == "windows":
    raise NotImplementedError("Temporarily not supported for Windows platform")
else:
    try:
        lib = ctypes.CDLL('search/c_mcts/mcts.so')
    except:
        ori_dir = os.getcwd()
        os.chdir("search/c_mcts")
        os.system("make clean")
        os.system("make")
        os.chdir(ori_dir)
        lib = ctypes.CDLL('search/c_mcts/mcts.so')
    c_mcts = lib.mcts_local_search
    c_mcts.argtypes = [
        ctypes.POINTER(ctypes.c_short),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),   
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float, 
        ctypes.c_int,                     
    ]
    c_mcts.restype = ctypes.POINTER(ctypes.c_int)
