import os

try:
    from .sources import cython_merge
except:
    ori_dir = os.getcwd()
    os.chdir("search/cython_merge/sources")
    os.system("python .\cython_merge_setup.py build_ext --inplace")
    os.chdir(ori_dir)    
    from .sources import cython_merge

merge_cython = cython_merge.merge_cython